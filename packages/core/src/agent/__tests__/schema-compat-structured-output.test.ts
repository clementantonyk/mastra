import { convertArrayToReadableStream, MockLanguageModelV2 } from '@internal/ai-sdk-v5/test';
import { describe, expect, it } from 'vitest';
import { z } from 'zod/v4';
import { Mastra } from '../../mastra';
import { Agent } from '../agent';

const HAIKU_MODEL_ID = 'claude-3.5-haiku-20241022';

function createStructuredOutputMockModel({
  provider,
  modelId,
  jsonText,
}: {
  provider: string;
  modelId: string;
  jsonText: string;
}) {
  return new MockLanguageModelV2({
    provider,
    modelId,
    doGenerate: async () => ({
      rawCall: { rawPrompt: null, rawSettings: {} },
      finishReason: 'stop',
      usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
      content: [{ type: 'text', text: jsonText }],
      warnings: [],
    }),
    doStream: async () => ({
      rawCall: { rawPrompt: null, rawSettings: {} },
      warnings: [],
      stream: convertArrayToReadableStream([
        { type: 'stream-start', warnings: [] },
        {
          type: 'response-metadata',
          id: 'id-0',
          modelId,
          timestamp: new Date(0),
        },
        { type: 'text-start', id: 'text-1' },
        { type: 'text-delta', id: 'text-1', delta: jsonText },
        { type: 'text-end', id: 'text-1' },
        {
          type: 'finish',
          finishReason: 'stop',
          usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
        },
      ]),
    }),
  });
}

describe('Agent structured output schema compat validation', () => {
  it('agent.generate accepts LLM JSON outside author numeric bounds on Anthropic Haiku', async () => {
    const schema = z.object({
      score: z.number().min(0).max(1),
    });
    const jsonText = JSON.stringify({ score: 1.2 });

    const agent = new Agent({
      id: 'structured-haiku-agent',
      name: 'Structured Haiku Agent',
      instructions: 'Return structured JSON only.',
      model: createStructuredOutputMockModel({
        provider: 'anthropic',
        modelId: HAIKU_MODEL_ID,
        jsonText,
      }),
    });

    const mastra = new Mastra({ agents: { agent }, logger: false });
    const result = await mastra.getAgent('agent').generate('Give a score', {
      structuredOutput: { schema },
    });

    expect(result.object).toEqual({ score: 1.2 });
  });

  it('agent.stream resolves object when LLM JSON violates author string min on Haiku', async () => {
    const schema = z.object({
      message: z.string().min(10),
    });
    const jsonText = JSON.stringify({ message: 'hi' });

    const agent = new Agent({
      id: 'structured-haiku-stream-agent',
      name: 'Structured Haiku Stream Agent',
      instructions: 'Return structured JSON only.',
      model: createStructuredOutputMockModel({
        provider: 'anthropic',
        modelId: HAIKU_MODEL_ID,
        jsonText,
      }),
    });

    const mastra = new Mastra({ agents: { agent }, logger: false });
    const stream = await mastra.getAgent('agent').stream('Short message', {
      structuredOutput: { schema },
    });

    const object = await stream.object;
    expect(object).toEqual({ message: 'hi' });
  });

  it('agent.generate still rejects out-of-schema JSON when compat layer does not apply', async () => {
    const schema = z.object({
      score: z.number().min(0).max(1),
    });
    const jsonText = JSON.stringify({ score: 1.2 });

    const agent = new Agent({
      id: 'structured-openai-agent',
      name: 'Structured OpenAI Agent',
      instructions: 'Return structured JSON only.',
      model: createStructuredOutputMockModel({
        provider: 'openai',
        modelId: 'gpt-4o-mini',
        jsonText,
      }),
    });

    const mastra = new Mastra({ agents: { agent }, logger: false });

    await expect(
      mastra.getAgent('agent').generate('Give a score', {
        structuredOutput: { schema },
      }),
    ).rejects.toThrow(/Structured output validation failed/i);
  });
});
