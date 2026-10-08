import type { StreamEvent } from './types'

// 网络数据块和 SSE 事件没有一一对应关系，因此保留未结束的行及事件。
export class SSEParser {
  private buffer = ''
  private dataLines: string[] = []
  private decoder = new TextDecoder()

  constructor(private readonly onEvent: (event: StreamEvent | '[DONE]') => void) {}

  feed(chunk: Uint8Array): void {
    // 流式解码可保留跨网络块的 UTF-8 字符，避免中文被截断成乱码。
    this.buffer += this.decoder.decode(chunk, { stream: true })
    this.consumeLines()
  }

  finish(): void {
    // 连接结束时也处理最后一个没有空行终止的事件。
    this.buffer += this.decoder.decode()
    this.consumeLines()
    if (this.buffer) this.consumeLine(this.buffer.replace(/\r$/, ''))
    this.buffer = ''
    this.dispatch()
  }

  private consumeLines(): void {
    let pos: number
    while ((pos = this.buffer.indexOf('\n')) !== -1) {
      this.consumeLine(this.buffer.slice(0, pos).replace(/\r$/, ''))
      this.buffer = this.buffer.slice(pos + 1)
    }
  }

  private consumeLine(line: string): void {
    // SSE 以空行结束一个事件；同一事件的多行 data 需合并。
    if (!line) {
      this.dispatch()
    } else if (line.startsWith('data:')) {
      this.dataLines.push(line.slice(5).trimStart())
    }
  }

  private dispatch(): void {
    if (!this.dataLines.length) return
    const data = this.dataLines.join('\n')
    this.dataLines = []
    if (data === '[DONE]') {
      // 后端使用 [DONE] 表示流结束，它不是 JSON 消息。
      this.onEvent('[DONE]')
      return
    }
    try {
      this.onEvent(JSON.parse(data) as StreamEvent)
    } catch {
      this.onEvent({ type: 'error', content: '无法解析服务端事件' })
    }
  }
}
