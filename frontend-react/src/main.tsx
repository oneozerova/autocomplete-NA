import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  loadWebModel,
  PromptAutocomplete,
  type WebModel,
} from "./features/prompt-autocomplete";
import "./App.css";

const DEMO_PROMPT =
  "нарисуй 4 девушки, синий фон, резкость 80%, поворот 45°, в тёмном лес";

function App() {
  const [model, setModel] = useState<WebModel | null>(null);
  const [promptA, setPromptA] = useState(DEMO_PROMPT);
  const [promptB, setPromptB] = useState(DEMO_PROMPT);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadWebModel("/web_model.json")
      .then(setModel)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <main className="app">
        <div className="app-card">
          <h1>Автодополнение промпта</h1>
          <p className="error">{error}</p>
          <p className="subtitle">
            Соберите модель: <code>python -m src.export_web</code>
          </p>
        </div>
      </main>
    );
  }

  if (!model) {
    return (
      <main className="app">
        <p className="loading">Загрузка модели…</p>
      </main>
    );
  }

  return (
    <main className="app">
      <div className="app-card">
        <span className="app-eyebrow">Вариант A — текущий</span>
        <h1>Промпт</h1>
        <p className="subtitle">
          Опишите картинку — подсказка появится серым. Параметры в тексте
          редактируются на месте. Угол — полукруг со шкалой.
        </p>
        <PromptAutocomplete
          value={promptA}
          onChange={setPromptA}
          model={model}
          label="Описание"
          placeholder="Например: нарисуй закат над горами, тёплый свет…"
          nextWordUrl="/next"
          angleChipStyle="minimal"
        />
        <div className="output">
          <span className="output-label">Итоговый промпт</span>
          <p className="output-text">{promptA || "—"}</p>
        </div>
      </div>

      <div className="app-card app-card--variant-b">
        <span className="app-eyebrow">Вариант B — эксперимент</span>
        <h1>Промпт</h1>
        <p className="subtitle">
          Цвет — названием, угол — орбита вокруг чипа. Минимум UI, те же
          параметры.
        </p>
        <PromptAutocomplete
          value={promptB}
          onChange={setPromptB}
          model={model}
          label="Описание"
          placeholder="Например: нарисуй закат над горами, тёплый свет…"
          colorChipStyle="label"
          angleChipStyle="protractor"
          nextWordUrl="/next"
        />
        <div className="output">
          <span className="output-label">Итоговый промпт</span>
          <p className="output-text">{promptB || "—"}</p>
        </div>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
