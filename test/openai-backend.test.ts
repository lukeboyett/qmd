/**
 * Tests for the OpenAI embedding backend.
 *
 * The `openai` module is mocked so these tests don't issue real API calls or
 * require a network connection. Each test configures the mock's
 * embeddings.create behavior and asserts on the number of calls and the
 * results produced by OpenAILLM.
 */
import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";

// Hoisted mock state. We declare it up here so the vi.mock factory below
// (which is hoisted above imports) can reach it. vitest hoists vi.mock, so
// we use vi.hoisted to make the state definition run in the same hoisted
// step.
const mockState = vi.hoisted(() => ({
  embedCalls: [] as Array<{ model: string; input: string | string[] }>,
  embedResponder: null as null | ((args: { model: string; input: string | string[] }) =>
    Promise<{ data: Array<{ index: number; embedding: number[] }>; model: string }>),
  reset() {
    this.embedCalls = [];
    this.embedResponder = null;
  },
}));

vi.mock("openai", () => {
  class FakeOpenAI {
    embeddings = {
      create: async (args: { model: string; input: string | string[] }) => {
        mockState.embedCalls.push(args);
        if (!mockState.embedResponder) {
          throw new Error("mockState.embedResponder not set");
        }
        return mockState.embedResponder(args);
      },
    };
    chat = { completions: { create: async () => ({ choices: [{ message: { content: "" } }], model: "gpt-4o-mini" }) } };
  }
  return { default: FakeOpenAI };
});

// Import *after* the mock is declared. These symbols resolve to the mocked
// module inside OpenAILLM's constructor.
import { OpenAILLM, getDefaultLlamaCpp, setDefaultLlamaCpp, LlamaCpp } from "../src/llm";

describe("OpenAI backend selection", () => {
  let originalKey: string | undefined;

  beforeEach(() => {
    originalKey = process.env.OPENAI_API_KEY;
    setDefaultLlamaCpp(null);
  });

  afterEach(() => {
    if (originalKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = originalKey;
    setDefaultLlamaCpp(null);
  });

  test("getDefaultLlamaCpp returns OpenAILLM when OPENAI_API_KEY is set", () => {
    process.env.OPENAI_API_KEY = "sk-test-fake";
    const llm = getDefaultLlamaCpp();
    expect(llm).toBeInstanceOf(OpenAILLM);
  });

  test("getDefaultLlamaCpp returns LlamaCpp when OPENAI_API_KEY is unset", () => {
    delete process.env.OPENAI_API_KEY;
    const llm = getDefaultLlamaCpp();
    expect(llm).toBeInstanceOf(LlamaCpp);
    expect(llm).not.toBeInstanceOf(OpenAILLM);
  });

  test("env selection wins over a pre-set singleton of the wrong backend", () => {
    // Simulates the CLI path where YAML `models:` config calls
    // setDefaultLlamaCpp(new LlamaCpp(...)) before OPENAI_API_KEY was
    // observed. The next getDefaultLlamaCpp() call must discard the stale
    // pre-set and return an OpenAILLM to match the current env.
    process.env.OPENAI_API_KEY = "sk-test";
    setDefaultLlamaCpp(new LlamaCpp({}));
    expect(getDefaultLlamaCpp()).toBeInstanceOf(OpenAILLM);
  });

  test("matching pre-set singleton is reused", () => {
    process.env.OPENAI_API_KEY = "sk-test";
    const preset = new OpenAILLM({ apiKey: "sk-test" });
    setDefaultLlamaCpp(preset);
    expect(getDefaultLlamaCpp()).toBe(preset);
  });

  test("OpenAILLM constructor throws when no API key is available", () => {
    delete process.env.OPENAI_API_KEY;
    expect(() => new OpenAILLM()).toThrow(/OPENAI_API_KEY/);
  });

  test("OpenAILLM ignores options.model and uses configured embed model", async () => {
    mockState.reset();
    mockState.embedResponder = async (args) => ({
      data: [{ index: 0, embedding: [0.1, 0.2, 0.3] }],
      model: String(args.model),
    });

    const llm = new OpenAILLM({ apiKey: "sk-test", embedModel: "text-embedding-3-large" });
    // Pass a local-GGUF-style model id on purpose; OpenAILLM must NOT forward it.
    const result = await llm.embed("hello world", { model: "hf:ggml-org/embeddinggemma-300M-GGUF/x.gguf" });

    expect(result?.embedding).toEqual([0.1, 0.2, 0.3]);
    expect(mockState.embedCalls).toHaveLength(1);
    expect(mockState.embedCalls[0].model).toBe("text-embedding-3-large");
  });
});

describe("OpenAI embedding retry + fallback", () => {
  beforeEach(() => {
    mockState.reset();
  });

  test("retries on transient 429 and eventually succeeds", async () => {
    let calls = 0;
    mockState.embedResponder = async () => {
      calls++;
      if (calls < 3) {
        const err: Error & { status?: number } = new Error("Too Many Requests");
        err.status = 429;
        throw err;
      }
      return {
        data: [{ index: 0, embedding: [1, 2, 3] }],
        model: "text-embedding-3-small",
      };
    };

    const llm = new OpenAILLM({ apiKey: "sk-test" });
    const result = await llm.embed("retry me");

    expect(result?.embedding).toEqual([1, 2, 3]);
    expect(calls).toBe(3);
  }, 30000);

  test("gives up after maxAttempts on persistent transient failures", async () => {
    mockState.embedResponder = async () => {
      const err: Error & { code?: string } = new Error("socket hang up");
      err.code = "ECONNRESET";
      throw err;
    };

    const llm = new OpenAILLM({ apiKey: "sk-test" });
    const result = await llm.embed("keeps failing");

    // embed() swallows the final error and returns null; mock was called
    // maxAttempts times (default 5 for single embed).
    expect(result).toBeNull();
    expect(mockState.embedCalls.length).toBe(5);
  }, 60000);

  test("does not retry on non-transient errors", async () => {
    let calls = 0;
    mockState.embedResponder = async () => {
      calls++;
      const err: Error & { status?: number } = new Error("Invalid API key");
      err.status = 401;
      throw err;
    };

    const llm = new OpenAILLM({ apiKey: "sk-test" });
    const result = await llm.embed("bad auth");

    expect(result).toBeNull();
    expect(calls).toBe(1);
  });

  test("embedBatch falls back to per-item embed when a batch fails repeatedly", async () => {
    // Fail any request whose input is an array (batch); succeed for single strings.
    mockState.embedResponder = async (args) => {
      if (Array.isArray(args.input)) {
        const err: Error & { status?: number } = new Error("Service Unavailable");
        err.status = 503;
        throw err;
      }
      return {
        data: [{ index: 0, embedding: [args.input.length] }],
        model: "text-embedding-3-small",
      };
    };

    const llm = new OpenAILLM({ apiKey: "sk-test" });
    const results = await llm.embedBatch(["alpha", "beta", "gamma"]);

    expect(results).toHaveLength(3);
    for (const r of results) {
      expect(r?.embedding).toHaveLength(1);
    }
    // Per-item embeds happened: at least 3 single-string calls after the batch
    // retries exhausted.
    const singleCalls = mockState.embedCalls.filter(c => typeof c.input === "string");
    expect(singleCalls.length).toBe(3);
  }, 60000);
});
