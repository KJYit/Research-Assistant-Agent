import { useRef, useState } from 'react';
import { GripHorizontal, Send } from 'lucide-react';

const MIN_HEIGHT = 52;
const MAX_HEIGHT = 160;
const SNAP_DISTANCE = 10;

interface ComposerProps {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}

export default function Composer({ value, loading, onChange, onSubmit, textareaRef }: ComposerProps) {
  const [height, setHeight] = useState<number | null>(null);
  const inkRef = useRef<HTMLSpanElement>(null);
  const drag = useRef<{ startY: number; height: number } | null>(null);
  const dragged = useRef(false);

  function resizeAt(pointerY: number, released: boolean) {
    if (!drag.current) return;
    const distance = drag.current.startY - pointerY;
    if (!dragged.current && Math.abs(distance) < 2) return;
    dragged.current = true;
    const next = Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, drag.current.height + distance));
    if (released && next <= MIN_HEIGHT + SNAP_DISTANCE) {
      setHeight(MIN_HEIGHT);
    } else if (released && next >= MAX_HEIGHT - SNAP_DISTANCE) {
      setHeight(MAX_HEIGHT);
    } else {
      setHeight(next);
    }
  }

  function updateText(next: string) {
    onChange(next);
    const ink = inkRef.current;
    if (!ink || next.length <= value.length || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    ink.getAnimations().forEach(animation => animation.cancel());
    ink.animate([
      { opacity: 0.65, transform: 'scale(0.45)', filter: 'blur(0)' },
      { opacity: 0, transform: 'scale(1.9)', filter: 'blur(2px)' },
    ], { duration: 420, easing: 'ease-out' });
  }

  return (
    <div className="input-area">
      <form className="input-form" onSubmit={event => { event.preventDefault(); onSubmit(); }}>
        <div className="composer-header">
          <label htmlFor="research-input">Your question</label>
          <button
            type="button"
            className="resize-handle"
            aria-label="Resize question box"
            aria-controls="research-input"
            title="Drag to resize or click to toggle the question box"
            onPointerDown={event => {
              if (!event.isPrimary || event.button !== 0) return;
              dragged.current = false;
              drag.current = { startY: event.clientY, height: textareaRef.current?.offsetHeight ?? MIN_HEIGHT };
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={event => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) resizeAt(event.clientY, false);
            }}
            onPointerUp={event => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) resizeAt(event.clientY, true);
            }}
            onLostPointerCapture={() => { drag.current = null; }}
            onClick={event => {
              if (event.detail !== 0 && dragged.current) return;
              const currentHeight = textareaRef.current?.offsetHeight ?? MIN_HEIGHT;
              setHeight(currentHeight >= MAX_HEIGHT ? MIN_HEIGHT : MAX_HEIGHT);
            }}
          >
            <GripHorizontal size={20} aria-hidden="true" />
          </button>
        </div>
        <div className="composer-body">
          <span className="input-ink" ref={inkRef} aria-hidden="true" />
          <textarea
            ref={textareaRef}
            id="research-input"
            className="input-field"
            placeholder="What would you like to explore?"
            value={value}
            onChange={event => updateText(event.target.value)}
            onKeyDown={event => {
              if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return;
              event.preventDefault();
              onSubmit();
            }}
            aria-describedby="composer-hint"
            disabled={loading}
            style={height === null ? undefined : { height }}
            rows={1}
          />
          <button id="btn-submit" type="submit" className="btn-submit" disabled={loading || !value.trim()}>
            <span>Send</span><Send size={16} aria-hidden="true" />
          </button>
        </div>
      </form>
      <p className="composer-hint" id="composer-hint">Enter to send · Shift + Enter for a new line</p>
    </div>
  );
}
