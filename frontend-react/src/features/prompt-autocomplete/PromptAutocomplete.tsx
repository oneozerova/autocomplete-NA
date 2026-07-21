import { useEffect, useRef } from "react";
import type { PromptAutocompleteProps } from "./types";
import { PromptController } from "./engine/promptController";
import "./PromptAutocomplete.css";

export function PromptAutocomplete({
  value,
  onChange,
  model,
  placeholder = "Опишите картинку…",
  label = "Промпт",
  className,
  disabled = false,
  nextWordUrl,
  showHint = true,
  colorChipStyle = "droplet",
  angleChipStyle = "protractor",
}: PromptAutocompleteProps) {
  const fieldRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<PromptController | null>(null);
  const onChangeRef = useRef(onChange);
  const lastEmitted = useRef<string | undefined>(value);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  useEffect(() => {
    const el = fieldRef.current;
    if (!el) return;

    const ctrl = new PromptController({
      el,
      model,
      nextWordUrl,
      colorChipStyle,
      angleChipStyle,
      onChange: (text) => {
        lastEmitted.current = text;
        onChangeRef.current?.(text);
      },
    });
    controllerRef.current = ctrl;

    if (value != null && value !== "") {
      ctrl.setText(value);
      lastEmitted.current = value;
    } else {
      ctrl.refresh(true);
    }

    return () => ctrl.destroy();
  }, [model, nextWordUrl, colorChipStyle, angleChipStyle]);

  useEffect(() => {
    if (value == null) return;
    if (value === lastEmitted.current) return;
    lastEmitted.current = value;
    controllerRef.current?.setText(value);
  }, [value]);

  useEffect(() => {
    controllerRef.current?.setDisabled(disabled);
  }, [disabled]);

  return (
    <div className={["pa-root", className].filter(Boolean).join(" ")}>
      {label ? (
        <div className="pa-header">
          <label className="pa-label">{label}</label>
          <span className="pa-badge">RU</span>
        </div>
      ) : null}

      <div className="pa-field-wrap">
        <div className="pa-field-glow" aria-hidden />
        <div
          ref={fieldRef}
          className="pa-prompt"
          contentEditable={!disabled}
          spellCheck={false}
          data-ph={placeholder}
          suppressContentEditableWarning
        />
      </div>

      {showHint ? (
        <div className="pa-footer">
          <span className="pa-tag">
            <kbd>Tab</kbd> принять
          </span>
          <span className="pa-tag pa-tag--muted">параметры кликабельны</span>
        </div>
      ) : null}
    </div>
  );
}
