# PURPOSE:
#   This file saves our data to Backblaze B2 — and loads it back
#   when we need it.
#   Think of B2 as a giant hard drive in the cloud that never
#   runs out of space and is always online (same idea as GCS,
#   just a different provider — and generally cheaper).

import json
import asyncio
from typing import Any


import boto3
# This is the official AWS Python library for talking to S3-shaped
# storage. Backblaze B2 speaks the same S3 API, so we point boto3
# at B2's endpoint instead of Amazon's and everything else works
# exactly the same way. Installed via:
#   pip install boto3

from botocore.exceptions import ClientError
# The specific exception type boto3 raises for "file not found",
# "access denied", etc. We use this to detect a missing file
# (the B2/S3 equivalent of GCS's 404).

from config.settings import settings
from config.logging_config import setup_logging
from ingestion.document_parser import ParsedStudy, ParsedPaper

logger = setup_logging(__name__)


# ─────────────────────────────────────────────────────────────
# FOLDER PATHS INSIDE OUR BUCKET
#
# A B2 bucket does not really have folders the way your
# computer does — but it LOOKS like it has folders because
# every file we save has a path-like name with slashes in it.
# Example: "raw/studies/NCT04788680.json" looks like a folder
# structure even though B2 technically just sees one long name
# (this is called the "key" in S3/B2 terminology, same idea as
# GCS's "blob name").
# Defining these as constants means if we ever want to change
# the folder layout, we only change it in ONE place.
# ─────────────────────────────────────────────────────────────

PREFIX_RAW_STUDIES       = "raw/studies"
PREFIX_RAW_PAPERS        = "raw/papers"
PREFIX_PROCESSED_STUDIES = "processed/studies"
PREFIX_PROCESSED_PAPERS  = "processed/papers"

class B2Store:
    """
    Handles saving data to and loading data from Backblaze B2.

    IMPORTANT — WHY WE USE asyncio.to_thread() THROUGHOUT THIS FILE:

    boto3 (the library we use to talk to B2) is "synchronous" —
    meaning when you ask it to upload a file, your whole program
    freezes and waits until the upload is done before doing
    anything else.

    But our entire MOSAIC system is built to be "asynchronous" —
    meaning we want our program to be able to do OTHER things
    while waiting for slow operations like uploads to finish.

    asyncio.to_thread() is the bridge between these two worlds.
    It takes a synchronous function (like a B2 upload) and runs
    it in a separate background thread, while letting our main
    program keep working on other tasks in the meantime.
    Think of it like handing a task to an assistant in another
    room, instead of standing there waiting yourself.
    """

    def __init__(self):
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.b2_endpoint_url,
            aws_access_key_id=settings.b2_key_id,
            aws_secret_access_key=settings.b2_application_key,
        )
        # Create a connection to Backblaze B2, disguised as an
        # S3 client. boto3 doesn't know or care that it's actually
        # talking to Backblaze — as long as we give it B2's
        # endpoint_url and B2's key pair instead of Amazon's,
        # every S3 method (put_object, get_object, list_objects...)
        # just works against our B2 bucket.
        # Like GCS's storage.Client(), this does NOT immediately
        # connect to the internet — it just sets up the object
        # that knows HOW to connect when we actually ask it to
        # do something.

        self._bucket_name = settings.b2_bucket_name
        # Unlike the GCS client (which gives you a bucket "object"
        # via self._client.bucket(...)), boto3's S3 client is
        # "bucket-less" — you just pass the bucket name as a string
        # into every call. We store it once here so we don't repeat
        # settings.b2_bucket_name everywhere below.

        logger.info(
            f"B2Store initialised | "
            f"bucket={settings.b2_bucket_name} | "
            f"endpoint={settings.b2_endpoint_url}"
        )

    async def save_raw_study(
        self,
        nct_id: str,
        data: dict[str, Any],
    ) -> str:
        """
        Saves the EXACT, untouched API response for one study.

        Call this the moment you receive data from the API —
        BEFORE any cleaning or parsing happens. This way, even
        if the parser has a bug, the original is always safe.

        Args:
            nct_id: The study's ID, used as the filename.
            data:   The raw study dictionary to save.

        Returns:
            The path inside B2 where the file was saved.
            Example: "raw/studies/NCT04788680.json"
        """

        b2_path = f"{PREFIX_RAW_STUDIES}/{nct_id}.json"
        # Build the full file path/name (called a "key" in S3/B2).
        # Example: "raw/studies/NCT04788680.json"

        await self._upload_json(path=b2_path, data=data)

        logger.info(f"Saved raw study | nct_id={nct_id} | path={b2_path}")
        return b2_path


    async def save_raw_paper(
        self,
        pmid: str,
        data: dict[str, Any],
    ) -> str:
        """
        Saves the EXACT, untouched data for one PubMed paper.
        Same idea as save_raw_study — but for papers.

        Args:
            pmid: The paper's PubMed ID, used as the filename.
            data: The raw paper dictionary to save.

        Returns:
            The path inside B2 where the file was saved.
        """

        b2_path = f"{PREFIX_RAW_PAPERS}/{pmid}.json"
        await self._upload_json(path=b2_path, data=data)

        logger.info(f"Saved raw paper | pmid={pmid} | path={b2_path}")
        return b2_path


    async def save_parsed_study(self, study: ParsedStudy) -> str:
        """
        Saves the CLEANED version of a study — after document_parser.py
        has already processed it into a ParsedStudy object.

        Args:
            study: A ParsedStudy object (the clean, typed version).

        Returns:
            The path inside B2 where the file was saved.
        """

        b2_path = f"{PREFIX_PROCESSED_STUDIES}/{study.nct_id}.json"

        await self._upload_json(
            path=b2_path,
            data=study.model_dump(),
        )

        logger.info(
            f"Saved parsed study | nct_id={study.nct_id} | path={b2_path}"
        )
        return b2_path
    

    async def save_parsed_paper(self, paper: ParsedPaper) -> str:
        """
        Saves the CLEANED version of a paper.

        Args:
            paper: A ParsedPaper object (the clean, typed version).

        Returns:
            The path inside B2 where the file was saved.
        """

        b2_path = f"{PREFIX_PROCESSED_PAPERS}/{paper.pmid}.json"

        await self._upload_json(
            path=b2_path,
            data=paper.model_dump(),
        )

        logger.info(
            f"Saved parsed paper | pmid={paper.pmid} | path={b2_path}"
        )
        return b2_path


    async def load_parsed_study(self, nct_id: str) -> ParsedStudy | None:
        """
        Loads a previously saved, cleaned study back from B2.

        This is the REVERSE of save_parsed_study — we use this
        in the processing layer when we need to read studies
        back in to chunk and embed them.

        Args:
            nct_id: Which study to load, by its NCT ID.

        Returns:
            A ParsedStudy object if found.
            None if no study with that ID exists in B2.
        """

        b2_path = f"{PREFIX_PROCESSED_STUDIES}/{nct_id}.json"
        data = await self._download_json(path=b2_path)
        # Download the raw JSON text and convert it back to a
        # Python dictionary (our private helper does this).

        if not data:
            return None
            # If nothing came back, the file probably does not exist.

        return ParsedStudy(**data)
        # **data "unpacks" the dictionary into keyword arguments.
        # Example: if data = {"nct_id": "NCT123", "title": "..."}
        # then ParsedStudy(**data) is the same as writing:
        # ParsedStudy(nct_id="NCT123", title="...")
        # This rebuilds our typed Pydantic object from the saved dict.


    async def list_processed_studies(self) -> list[str]:
        """
        Returns a list of every study's NCT ID currently saved
        in the "processed" folder of our bucket.

        We use this later in the processing layer to know exactly
        which studies are available to chunk and embed, without
        needing to ask the database first.

        Returns:
            A list of NCT ID strings.
            Example: ["NCT04788680", "NCT03232294", "NCT06796322"]
        """

        nct_ids: list[str] = []
        continuation_token = None
        # S3/B2's list_objects_v2 only returns up to 1000 keys per
        # call. If a bucket has more files than that, it hands back
        # a "continuation_token" meaning "there's more — call me
        # again with this token to get the next page". GCS's
        # list_blobs() hides this pagination from you automatically;
        # with boto3 we have to loop through the pages ourselves.

        while True:
            list_kwargs = {
                "Bucket": self._bucket_name,
                "Prefix": PREFIX_PROCESSED_STUDIES,
                # Prefix= means: only show files whose path starts with
                # "processed/studies" — i.e. only the files we want.
            }
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = await asyncio.to_thread(
                self._client.list_objects_v2,
                **list_kwargs,
            )

            for obj in response.get("Contents", []):
                key = obj["Key"]

                filename = key.split("/")[-1]

                nct_id = filename.replace(".json", "")
    
                if nct_id:
                    nct_ids.append(nct_id)

            if response.get("IsTruncated"):
                continuation_token = response.get("NextContinuationToken")
            else:
                break

        logger.info(f"Listed processed studies | count={len(nct_ids)}")
        return nct_ids


    async def _upload_json(
        self,
        path: str,
        data: dict[str, Any],
    ) -> None:
        """
        The shared internal method that ACTUALLY does the uploading.
        Every save_* method above eventually calls this one.

        Args:
            path: The destination path inside the B2 bucket.
            data: The dictionary to save as JSON.
        """

        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket_name,
            Key=path,
            Body=json_bytes,
            ContentType="application/json",
        )


    async def _download_json(
        self,
        path: str,
    ) -> dict[str, Any] | None:
        """
        The shared internal method that downloads a file from B2
        and converts it back into a Python dictionary.

        Args:
            path: The path inside the B2 bucket to download from.

        Returns:
            A Python dictionary if the file was found.
            None if the file does not exist or something went wrong.
        """

        try:
            response = await asyncio.to_thread(
                self._client.get_object,
                Bucket=self._bucket_name,
                Key=path,
            )
            # get_object is the S3/B2 equivalent of GCS's
            # blob.download_as_bytes(). Again wrapped in
            # asyncio.to_thread() since boto3 calls are synchronous
            # by default.

            json_bytes = await asyncio.to_thread(response["Body"].read)
            # Unlike GCS, boto3 doesn't hand you the bytes directly —
            # response["Body"] is a streaming file-like object, so we
            # need one more call, .read(), to pull the actual bytes
            # out of it. We still run this in a thread since it's
            # still a blocking network read under the hood.

            return json.loads(json_bytes.decode("utf-8"))

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")

            if error_code == "NoSuchKey":
                logger.warning(f"File not found in B2 | path={path}")
            else:
                logger.error(
                    f"Failed to download from B2 | path={path} | error={e}"
                )
            return None

        except Exception as e:
            logger.error(
                f"Failed to download from B2 | path={path} | error={e}"
            )
            return None