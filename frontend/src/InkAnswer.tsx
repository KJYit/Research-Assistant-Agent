import { useEffect, useRef, useState } from 'react';
import type { Element, Root, RootContent } from 'hast';
import { BookOpen } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const GLYPH_INTERVAL = 30;
const FOCUS_DURATION = 120;
const REDUCED_MOTION = '(prefers-reduced-motion: reduce)';
const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });

function rehypeInk() {
  function mark(node: RootContent): RootContent[] {
    if (node.type === 'element') {
      if (node.tagName === 'pre' || node.tagName === 'code') {
        node.properties.dataInkGlyph = '';
      } else {
        node.children = node.children.flatMap(mark) as Element['children'];
      }
    }
    if (node.type !== 'text') return [node];

    return Array.from(segmenter.segment(node.value), ({ segment }): RootContent =>
      segment.trim() ? {
        type: 'element',
        tagName: 'span',
        properties: { dataInkGlyph: '' },
        children: [{ type: 'text', value: segment }],
      } : { type: 'text', value: segment });
  }

  return (tree: Root) => { tree.children = tree.children.flatMap(mark); };
}

export default function InkAnswer({ content, animate }: { content: string; animate: boolean }) {
  const [revealing, setRevealing] = useState(() => animate && !matchMedia(REDUCED_MOTION).matches);
  const bodyRef = useRef<HTMLDivElement>(null);
  const skipRef = useRef<HTMLButtonElement>(null);
  const active = animate && revealing;

  useEffect(() => {
    const body = bodyRef.current;
    if (!body || !active) return;

    const glyphs = body.querySelectorAll<HTMLElement>('[data-ink-glyph]');
    const preference = matchMedia(REDUCED_MOTION);
    let frame = 0;
    let started: number | undefined;
    let next = 0;

    function finish() {
      if (document.activeElement === skipRef.current) body?.focus({ preventScroll: true });
      setRevealing(false);
    }

    function draw(now: number) {
      started ??= now;
      const elapsed = now - started;
      while (next < glyphs.length && elapsed >= next * GLYPH_INTERVAL) {
        const glyph = glyphs[next];
        const age = elapsed - next * GLYPH_INTERVAL;
        glyph.dataset.inkReady = '';
        if (age < FOCUS_DURATION) {
          const animation = glyph.animate([
            { opacity: 0, filter: 'blur(6px)' },
            { opacity: 1, filter: 'blur(0)' },
          ], { duration: FOCUS_DURATION, easing: 'ease-out' });
          animation.currentTime = age;
        }
        next += 1;
      }
      if (elapsed < (glyphs.length - 1) * GLYPH_INTERVAL + FOCUS_DURATION) {
        frame = requestAnimationFrame(draw);
      } else {
        finish();
      }
    }

    preference.addEventListener('change', finish);
    frame = requestAnimationFrame(draw);
    return () => {
      preference.removeEventListener('change', finish);
      cancelAnimationFrame(frame);
      glyphs.forEach(glyph => glyph.getAnimations().forEach(animation => animation.cancel()));
    };
  }, [active, content]);

  return (
    <>
      <div className="answer-heading">
        <h2 className="answer-header"><BookOpen size={17} aria-hidden="true" /> Research answer</h2>
        {active && <button type="button" className="show-answer" ref={skipRef} onClick={() => {
          bodyRef.current?.focus({ preventScroll: true });
          setRevealing(false);
        }}>Show full answer</button>}
      </div>
      <div className="answer-body" data-revealing={active || undefined} ref={bodyRef} tabIndex={-1}>
        <ReactMarkdown rehypePlugins={active ? [rehypeInk] : []}
          disallowedElements={['a', 'img']} unwrapDisallowed>
          {content}
        </ReactMarkdown>
      </div>
    </>
  );
}
