# React frontend — встраиваемый компонент

Переиспользуемый React-компонент поля промпта с offline ghost-подсказками и инлайн-чипами (числа, %, цвета, углы, пресеты). Порт логики из `frontend/prompt.html`.

## Быстрый старт

```bash
# из корня репозитория — если модели ещё нет
python -m src.export_web

cd frontend-react
npm install
npm run dev
```

Откройте http://localhost:5173

## Использование в своём React-проекте

Скопируйте папку `src/features/prompt-autocomplete` в ваш проект или подключите как workspace-пакет.

```tsx
import {
  PromptAutocomplete,
  loadWebModel,
} from "@/features/prompt-autocomplete";

const model = await loadWebModel("/static/web_model.json");

<PromptAutocomplete
  value={prompt}
  onChange={setPrompt}
  model={model}
  nextWordUrl="/api/prompt/next"   // опционально: LLM next-word
/>
```

## Props

| Prop | Описание |
|------|----------|
| `model` | `web_model.json` — обязателен |
| `value` / `onChange` | controlled-режим |
| `placeholder` | текст-подсказка в пустом поле |
| `nextWordUrl` | POST endpoint для LLM (как `scripts/serve.py` → `/next`) |
| `disabled` | отключить ввод |

## Структура

```
src/features/prompt-autocomplete/
  engine/
    ghostEngine.ts      # offline дополнение окончаний
    chipDetect.ts       # распознавание параметров в тексте
    chipDom.ts          # DOM-фабрики чипов
    domUtils.ts         # serialize / caret
    promptController.ts # оркестрация (как в prompt.html)
  PromptAutocomplete.tsx
  PromptAutocomplete.css
  index.ts
```
