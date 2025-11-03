import ollama from "npm:ollama";
import { z } from "npm:zod";
import { zodToJsonSchema } from "npm:zod-to-json-schema";
import OpenAI from "npm:openai";

export const OLLAMA_URL = "http://127.0.0.1:11434/api/chat";
export type Msg = { role: "system"|"user"|"assistant"; content: string };
export type ChatRes<T> = { data: T|null; raw: string; tokens: number };

export type LLMProvider = "ollama" | "openai";

export interface LLMConfig {
  provider: LLMProvider;
  model: string;
  apiKey?: string; // openai key
  baseURL?: string; // endpoints
}


export async function chatJSON<T>(
  config: LLMConfig,
  messages: Msg[],
  schema: z.ZodType<T>,
): Promise<ChatRes<T>> {
    if (config.provider === "ollama") {
    return chatJSONOllama(config.model, messages, schema);
  } else if (config.provider === "openai") {
    return chatJSONOpenAI(config, messages, schema);
  }
    throw new Error(`Unsupported LLM provider: ${config.provider}`);
}

async function chatJSONOllama<T>(
  model: string,
  messages: Msg[],
  schema: z.ZodType<T>
): Promise<ChatRes<T>> {
  const resp = await ollama.chat({
    model,
    messages,
    stream: false,
    format: zodToJsonSchema(schema),
    options: { temperature: 0.2 },
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

    if (
      (cleaned.startsWith('"') && cleaned.endsWith('"')) ||
      (cleaned.startsWith("'") && cleaned.endsWith("'"))
    ) {
      const inner = cleaned.slice(1, -1);
      if (
        (inner.includes("{") && inner.includes("}")) ||
        (inner.includes("[") && inner.includes("]"))
      ) {
        cleaned = inner
          .replace(/\\"/g, '"')
          .replace(/\\'/g, "'")
          .replace(/\\\\/g, "\\");
      }
    }

    cleaned = cleaned.replace(/,(\s*[}\]])/g, "$1");
    const parsed = JSON.parse(cleaned);
    data = schema.parse(parsed);
  } catch (error) {
    if (globalThis.Deno?.env?.get("DEBUG")) {
      console.error("JSON parsing failed:", error.message);
    }
  }

  const tokens =
    (resp?.prompt_eval_count ?? 0) + (resp?.eval_count ?? 0) ||
    Math.ceil((JSON.stringify(messages).length + rawOriginal.length) / 4);

  return { data, raw: rawOriginal, tokens };
}

async function chatJSONOpenAI<T>(
  config: LLMConfig,
  messages: Msg[],
  schema: z.ZodType<T>
): Promise<ChatRes<T>> {
  const client = new OpenAI({
    apiKey: config.apiKey || Deno.env.get("OPENAI_API_KEY"),
    baseURL: config.baseURL,
  });

  // Convert messages format
  const openaiMessages = messages.map((m) => ({
    role: m.role as "system" | "user" | "assistant",
    content: m.content,
  }));

  try {
    // Use JSON mode for OpenAI
    const completion = await client.chat.completions.create({
      model: config.model,
      messages: openaiMessages,
      temperature: 0.2,
      response_format: { type: "json_object" },
    });

    const rawContent = completion.choices[0].message.content || "";
    let data: T | null = null;

    try {
      const parsed = JSON.parse(rawContent);
      data = schema.parse(parsed);
    } catch (error) {
      if (globalThis.Deno?.env?.get("DEBUG")) {
        console.error("OpenAI JSON parsing failed:", error.message);
      }
    }

    // Token counting for OpenAI
    const tokens = completion.usage?.total_tokens ||
      Math.ceil((JSON.stringify(messages).length + rawContent.length) / 4);

    return { data, raw: rawContent, tokens };
  } catch (error) {
    console.error("OpenAI API error:", error);
    return { data: null, raw: "", tokens: 0 };
  }
}

export async function chatJSONWithRetry<T>(
  config: LLMConfig,
  messages: Msg[],
  schema: z.ZodType<T>,
  maxRetries = 3
): Promise<ChatRes<T>> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const result = await chatJSON(config, messages, schema);

      // parsing succeeded
      if (result.data !== null) {
        return result;
      }

      // parsing failed but not an error
      if (result.raw !== "") {
        console.warn(`Attempt ${attempt + 1}: Got response but parsing failed`);
      }
    } catch (error) {
      lastError = error as Error;
      console.warn(`Attempt ${attempt + 1} failed:`, error.message);
    }

    // If not the last attempt, wait before retrying
    if (attempt < maxRetries - 1) {
      const delay = Math.pow(2, attempt) * 1000;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  console.error(`All ${maxRetries} attempts failed. Last error:`, lastError);
  return { data: null, raw: "", tokens: 0 };
}
