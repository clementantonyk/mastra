import type { JSONSchema7, Schema } from '@internal/ai-sdk-v5';
import { AnthropicSchemaCompatLayer, applyCompatLayer, isZodType, toStandardSchema } from '@mastra/schema-compat';
import type { z as z3 } from 'zod/v3';
import type { z as z4 } from 'zod/v4';
import type { PublicSchema, StandardSchemaWithJSON } from '../../schema';
import { isStandardSchemaWithJSON, standardSchemaToJSONSchema } from '../../schema';

export type PartialSchemaOutput<OUTPUT = undefined> = OUTPUT extends undefined ? undefined : Partial<OUTPUT>;

/**
 * @deprecated Use StandardSchemaWithJSON from '../../schema' instead
 */
export type OutputSchema<OBJECT = any> =
  | z4.ZodType<OBJECT, any>
  | z3.Schema<OBJECT, z3.ZodTypeDef, any>
  | Schema<OBJECT>
  | JSONSchema7
  | undefined;

/**
 * @deprecated Use StandardSchemaWithJSON from '../../schema' instead
 * Legacy type for schema validation.
 */
export type SchemaWithValidation<T = any> = z4.ZodType<T, any> | z3.Schema<T, z3.ZodTypeDef, any>;

/**
 * @deprecated Use InferPublicSchema or InferStandardSchemaOutput from '../../schema' instead
 * Infer the output type from a schema
 */
export type InferSchemaOutput<T> =
  T extends z4.ZodType<infer O, any>
    ? O
    : T extends z3.Schema<infer O, z3.ZodTypeDef, any>
      ? O
      : T extends Schema<infer O>
        ? O
        : unknown;

/**
 * @deprecated Use PublicSchema from '../../schema' instead
 */
export type InferZodLikeSchema<T> =
  T extends z4.ZodType<infer O, any> ? O : T extends z3.Schema<infer O, z3.ZodTypeDef, any> ? O : unknown;

export type ZodLikePartialSchema<T = any> =
  | (z4.core.$ZodType<Partial<T>, any> & {
      safeParse(value: unknown): { success: boolean; data?: Partial<T>; error?: any };
    })
  | (z3.ZodType<Partial<T>, z3.ZodTypeDef, any> & {
      safeParse(value: unknown): { success: boolean; data?: Partial<T>; error?: any };
    });

export function asJsonSchema(schema: StandardSchemaWithJSON | undefined): JSONSchema7 | undefined {
  if (!schema) {
    return undefined;
  }

  // Handle StandardSchemaWithJSON
  if (isStandardSchemaWithJSON(schema)) {
    // Use 'input' IO mode to get the schema BEFORE transforms are applied
    // This is critical for OpenAI compat transforms that add .transform()
    // which can't be properly represented in JSON Schema
    //
    // Use 'draft-07' target for maximum compatibility with LLM providers
    const jsonSchema = standardSchemaToJSONSchema(schema, { io: 'input', target: 'draft-07' });

    return jsonSchema;
  }

  return schema;
}

export type SchemaModelInfo = {
  provider: string;
  modelId: string;
  supportsStructuredOutputs: boolean;
};

/** Same compat layers as `getResponseFormat` / `getTransformedSchema` (Anthropic when model is set). */
export function getStructuredOutputCompatLayers(model?: SchemaModelInfo): AnthropicSchemaCompatLayer[] {
  if (!model) {
    return [];
  }
  return [new AnthropicSchemaCompatLayer(model)];
}

/** JSON Schema used for final validation — matches unwrapped value shape handlers validate. */
function structuredOutputValidationJsonSchema(
  transformed: NonNullable<ReturnType<typeof getTransformedSchema>>,
): JSONSchema7 {
  if (transformed.outputFormat === 'array') {
    const elements = transformed.jsonSchema.properties?.elements as JSONSchema7 | undefined;
    const items = elements?.items ?? elements;
    return { type: 'array', items: items as JSONSchema7 | JSONSchema7[] | undefined };
  }

  if (transformed.outputFormat === 'enum') {
    const result = transformed.jsonSchema.properties?.result as JSONSchema7 | undefined;
    if (result) {
      return result;
    }
  }

  return transformed.jsonSchema;
}

/**
 * Schema used when validating parsed structured output — must match the compat-transformed
 * JSON the model saw in `getResponseFormat`, not the author's original constraints.
 */
export function buildStructuredOutputValidationSchema<OUTPUT>(
  schema: PublicSchema<OUTPUT> | undefined,
  options?: { model?: SchemaModelInfo },
): StandardSchemaWithJSON<OUTPUT> | undefined {
  if (!schema) {
    return undefined;
  }

  let resolved: unknown = schema;
  if (typeof resolved === 'function') {
    resolved = (resolved as () => unknown)();
  }

  const normalized = isZodType(resolved)
    ? (toStandardSchema(resolved) as StandardSchemaWithJSON<OUTPUT>)
    : (toStandardSchema(resolved as PublicSchema<OUTPUT>) as StandardSchemaWithJSON<OUTPUT>);

  const compatLayers = getStructuredOutputCompatLayers(options?.model);
  const compatApplies = compatLayers.some(layer => layer.shouldApply());

  if (compatApplies) {
    const transformed = getTransformedSchema(normalized, options);
    if (transformed?.jsonSchema) {
      const validationJson = structuredOutputValidationJsonSchema(transformed);
      return toStandardSchema(validationJson) as StandardSchemaWithJSON<OUTPUT>;
    }
  }

  return normalized;
}

export function getTransformedSchema<OUTPUT = undefined>(
  schema?: StandardSchemaWithJSON<OUTPUT>,
  options?: { model?: SchemaModelInfo },
) {
  if (!schema) {
    return undefined;
  }

  const compatLayers = getStructuredOutputCompatLayers(options?.model);
  const jsonSchema =
    compatLayers.length > 0
      ? (applyCompatLayer({
          schema: schema as PublicSchema<OUTPUT>,
          compatLayers,
          mode: 'jsonSchema',
        }) as JSONSchema7)
      : asJsonSchema(schema);

  if (!jsonSchema) {
    return undefined;
  }

  const { $schema, ...itemSchema } = jsonSchema;
  if (itemSchema.type === 'array') {
    const innerElement = itemSchema.items;
    const arrayOutputSchema: JSONSchema7 = {
      $schema: $schema,
      type: 'object',
      properties: {
        elements: { type: 'array', items: innerElement },
      },
      required: ['elements'],
      additionalProperties: false,
    };

    return {
      jsonSchema: arrayOutputSchema,
      outputFormat: 'array',
    };
  }

  // Handle enum schemas - wrap in object like AI SDK does
  if (itemSchema.enum && Array.isArray(itemSchema.enum)) {
    const enumOutputSchema: JSONSchema7 = {
      $schema: $schema,
      type: 'object',
      properties: {
        result: { type: itemSchema.type || 'string', enum: itemSchema.enum },
      },
      required: ['result'],
      additionalProperties: false,
    };

    return {
      jsonSchema: enumOutputSchema,
      outputFormat: 'enum',
    };
  }

  return {
    jsonSchema: jsonSchema,
    outputFormat: jsonSchema.type, // 'object'
  };
}

export function getResponseFormat(
  schema?: StandardSchemaWithJSON,
  options?: { model?: SchemaModelInfo },
):
  | {
      type: 'text';
    }
  | {
      type: 'json';
      /**
       * JSON schema that the generated output should conform to.
       */
      schema?: JSONSchema7;
    } {
  if (schema) {
    const transformedSchema = getTransformedSchema(schema, options);
    return {
      type: 'json',
      schema: transformedSchema?.jsonSchema,
    };
  }

  // response format 'text' for everything else
  return {
    type: 'text',
  };
}
