import { describe, expect, it } from 'vitest'
import { SSEParser } from './sse'
import type { StreamEvent } from './types'

describe('SSEParser', () => {
  it('restores UTF-8 events across arbitrary network chunks', () => {
    const events: (StreamEvent | '[DONE]')[] = []
    const parser = new SSEParser(event => events.push(event))
    const bytes = new TextEncoder().encode('data: {"type":"token","content":"论文"}\r\n\r\ndata: [DONE]\n\n')
    for (const byte of bytes) parser.feed(Uint8Array.of(byte))
    parser.finish()
    expect(events).toEqual([{ type: 'token', content: '论文' }, '[DONE]'])
  })

  it('flushes an event when the stream ends without a blank line', () => {
    const events: (StreamEvent | '[DONE]')[] = []
    const parser = new SSEParser(event => events.push(event))
    parser.feed(new TextEncoder().encode('data: {"type":"error","content":"失败"}'))
    parser.finish()
    expect(events).toEqual([{ type: 'error', content: '失败' }])
  })
})
