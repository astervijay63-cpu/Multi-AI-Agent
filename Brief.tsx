import type { ReactNode } from "react";

// The supervisor asks GPT to write the brief in a light markdown dialect —
// **bold** headers, numbered items, dashed bullets. That is a small enough
// subset to render directly, so the app ships no markdown dependency.
// Anything unrecognised falls through as a plain paragraph rather than
// being dropped, so nothing the model writes ever disappears from view.

function inline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /\*\*(.+?)\*\*|`(.+?)`|\b(NCT\d{8})\b/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      nodes.push(text.slice(cursor, match.index));
    }
    const key = `${keyPrefix}-${match.index}`;
    if (match[1] !== undefined) {
      nodes.push(<strong key={key}>{match[1]}</strong>);
    } else if (match[2] !== undefined) {
      nodes.push(<code key={key}>{match[2]}</code>);
    } else {
      nodes.push(
        <span className="nct" key={key}>
          {match[3]}
        </span>,
      );
    }
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) nodes.push(text.slice(cursor));
  return nodes;
}

export function Brief({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  const lines = text.split("\n");
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushList = () => {
    if (!list) return;
    const items = list.items.map((item, i) => (
      <li key={i}>{inline(item, `li-${blocks.length}-${i}`)}</li>
    ));
    blocks.push(
      list.ordered ? (
        <ol key={blocks.length}>{items}</ol>
      ) : (
        <ul key={blocks.length}>{items}</ul>
      ),
    );
    list = null;
  };

  for (const raw of lines) {
    const line = raw.trim();

    if (!line) {
      flushList();
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flushList();
      const level = Math.min(heading[1].length + 1, 6);
      const Tag = `h${level}` as "h2" | "h3" | "h4" | "h5" | "h6";
      blocks.push(<Tag key={blocks.length}>{inline(heading[2], "h")}</Tag>);
      continue;
    }

    const ordered = /^(\d+)[.)]\s+(.*)$/.exec(line);
    if (ordered) {
      if (!list?.ordered) {
        flushList();
        list = { ordered: true, items: [] };
      }
      list.items.push(ordered[2]);
      continue;
    }

    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    if (bullet) {
      if (list && !list.ordered) {
        list.items.push(bullet[1]);
      } else {
        flushList();
        list = { ordered: false, items: [bullet[1]] };
      }
      continue;
    }

    // A line that is entirely bold reads as a section header in these
    // briefs, so promote it instead of rendering a one-word paragraph.
    const boldHeader = /^\*\*(.+?):?\*\*:?$/.exec(line);
    if (boldHeader) {
      flushList();
      blocks.push(<h3 key={blocks.length}>{boldHeader[1]}</h3>);
      continue;
    }

    flushList();
    blocks.push(<p key={blocks.length}>{inline(line, `p${blocks.length}`)}</p>);
  }

  flushList();
  return <div className="brief">{blocks}</div>;
}
