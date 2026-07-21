export { PromptAutocomplete } from "./PromptAutocomplete";
export type {
  PromptAutocompleteProps,
  WebModel,
  NextWordResponse,
} from "./types";
export { GhostEngine, shouldSuggestNextWord } from "./engine/ghostEngine";
export { detectChips } from "./engine/chipDetect";

export async function loadWebModel(url: string): Promise<import("./types").WebModel> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load model: ${res.status}`);
  return res.json();
}
