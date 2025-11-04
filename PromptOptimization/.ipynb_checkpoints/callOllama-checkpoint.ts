import ollama from "npm:ollama";
import { z } from "npm:zod";
import { zodToJsonSchema } from "npm:zod-to-json-schema";

export const OLLAMA_URL = "http://127.0.0.1:11434/api/chat";
export type Msg = { role: "system"|"user"|"assistant"; content: string };
export type ChatRes<T> = { data: T|null; raw: string; tokens: number };

export async function chatJSON<T>(
  model: string,
  messages: Msg[],
  schema: z.ZodType<T>,
): Promise<ChatRes<T>> {
  const resp = await ollama.chat({
    model,
    messages,
    stream: false,
    format: zodToJsonSchema(schema),
    options: { 
      temperature: 0.2,
    //   num_predict: -1  // Prevent truncation
    }
  });
  
  const rawOriginal = resp?.message?.content ?? "";
  let data: T | null = null;
  
  try {
    let cleaned = rawOriginal.trim();
    
    cleaned = cleaned
      .replace(/[\u201C\u201D\u201E\u201F\u2033\u2036]/g, '"')
      .replace(/[\u2018\u2019\u201A\u201B\u2032\u2035]/g, "'")
      .replace(/[""]/g, '"')
      .replace(/['']/g, "'")
      .replace(/[``]/g, '"');
    
    if ((cleaned.startsWith('"') && cleaned.endsWith('"')) ||
        (cleaned.startsWith("'") && cleaned.endsWith("'"))) {
      
      const inner = cleaned.slice(1, -1);
      
      if ((inner.includes('{') && inner.includes('}')) ||
          (inner.includes('[') && inner.includes(']'))) {
        cleaned = inner
          .replace(/\\"/g, '"')
          .replace(/\\'/g, "'")
          .replace(/\\\\/g, '\\');
      }
    }
    
    // Step 3: Clean trailing commas
    cleaned = cleaned.replace(/,(\s*[}\]])/g, '$1');
    
    // Step 4: Try to parse
    const parsed = JSON.parse(cleaned);
    data = schema.parse(parsed);
    
  } catch (error) {
    // Silent fail - let caller handle null data
    // Only log in development/debug mode
    if (globalThis.Deno?.env?.get("DEBUG")) {
      console.error("JSON parsing failed:");
      console.error("  Original:", rawOriginal);
      console.error("  Error:", error.message);
    }
  }
  
  const tokens = 
    (resp?.prompt_eval_count ?? 0) + 
    (resp?.eval_count ?? 0) || 
    Math.ceil((JSON.stringify(messages).length + rawOriginal.length) / 4);
  
  return { data, raw: rawOriginal, tokens };
}