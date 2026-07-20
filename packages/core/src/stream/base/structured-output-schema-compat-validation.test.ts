import { describe, expect, it } from 'vitest';
import { z } from 'zod/v4';
import { convertArrayToReadableStream, convertAsyncIterableToArray } from '../../loop/test-utils/stream-helpers';
import { ChunkFrom } from '../types';
import type { ChunkType } from '../types';
import { createObjectStreamTransformer } from './output-format-handlers';
import type { SchemaModelInfo } from './schema';

const haikuModel: SchemaModelInfo = {
  provider: 'anthropic',
  modelId: 'claude-3.5-haiku-20241022',
  supportsStructuredOutputs: true,
};

describe('structured output schema compat validation', () => {
  it('accepts JSON that violates author numeric bounds when Anthropic compat applies', async () => {
    const schema = z.object({
      score: z.number().min(0).max(1),
    });

    const transformer = createObjectStreamTransformer({
      structuredOutput: { schema },
      model: haikuModel,
    });

    const streamParts: ChunkType<typeof schema>[] = [
      {
        type: 'text-delta',
        runId: 'test-run',
        from: ChunkFrom.AGENT,
        payload: { id: 'text-1', text: '{"score":1.2}' },
      },
      {
        type: 'text-end',
        runId: 'test-run',
        from: ChunkFrom.AGENT,
        payload: { id: 'text-1' },
      },
    ];

    // @ts-expect-error - web/stream readable stream type error
    const stream = convertArrayToReadableStream(streamParts).pipeThrough(transformer);
    const chunks = await convertAsyncIterableToArray(stream);

    const errorChunk = chunks.find(c => c?.type === 'error');
    expect(errorChunk).toBeUndefined();

    const objectResultChunk = chunks.find(c => c?.type === 'object-result');
    expect(objectResultChunk?.object).toEqual({ score: 1.2 });
  });

  it('still validates with author schema when no model is provided', async () => {
    const schema = z.object({
      score: z.number().min(0).max(1),
    });

    const transformer = createObjectStreamTransformer({
      structuredOutput: { schema },
    });

    const streamParts: ChunkType<typeof schema>[] = [
      {
        type: 'text-delta',
        runId: 'test-run',
        from: ChunkFrom.AGENT,
        payload: { id: 'text-1', text: '{"score":1.2}' },
      },
      {
        type: 'text-end',
        runId: 'test-run',
        from: ChunkFrom.AGENT,
        payload: { id: 'text-1' },
      },
    ];

    // @ts-expect-error - web/stream readable stream type error
    const stream = convertArrayToReadableStream(streamParts).pipeThrough(transformer);
    const chunks = await convertAsyncIterableToArray(stream);

    const errorChunk = chunks.find(c => c?.type === 'error');
    expect(errorChunk).toBeDefined();
    expect((errorChunk?.payload?.error as Error).message).toContain('Structured output validation failed');
  });

  it('accepts short strings when Haiku strips minLength from compat schema', async () => {
    const schema = z.object({
      message: z.string().min(10),
    });

    const transformer = createObjectStreamTransformer({
      structuredOutput: { schema },
      model: haikuModel,
    });

    const streamParts: ChunkType<typeof schema>[] = [
      {
        type: 'text-delta',
        runId: 'test-run',
        from: ChunkFrom.AGENT,
        payload: { id: 'text-1', text: '{"message":"hi"}' },
      },
      {
        type: 'text-end',
        runId: 'test-run',
        from: ChunkFrom.AGENT,
        payload: { id: 'text-1' },
      },
    ];

    // @ts-expect-error - web/stream readable stream type error
    const stream = convertArrayToReadableStream(streamParts).pipeThrough(transformer);
    const chunks = await convertAsyncIterableToArray(stream);

    expect(chunks.find(c => c?.type === 'error')).toBeUndefined();
    expect(chunks.find(c => c?.type === 'object-result')?.object).toEqual({ message: 'hi' });
  });
});
