// Package aimix calls OpenAI and Anthropic models through one streaming API.
//
// The root package holds the provider-neutral types: Model, Context, the
// message and content types, and the AssistantMessageEvent stream. The api
// package implements each wire protocol (anthropic-messages, openai-responses,
// openai-completions) on top of the official SDKs, and the providers package
// defines the providers that resolve API keys and base URLs. Client routes a
// request to the provider named by Model.Provider.
package aimix
