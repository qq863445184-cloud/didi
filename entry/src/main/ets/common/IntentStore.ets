import { Want } from '@kit.AbilityKit'

export interface LaunchIntent {
  id: string
  intentType: string
  requestId: string
  driverId: string
}

class IntentMemoryStore {
  private latest: LaunchIntent = {
    id: '',
    intentType: '',
    requestId: '',
    driverId: ''
  }

  updateFromWant(want: Want): void {
    const params: Record<string, Object> = (want.parameters ?? {}) as Record<string, Object>
    const now: string = Date.now().toString()
    this.latest = {
      id: now,
      intentType: String(params['intentType'] ?? ''),
      requestId: String(params['requestId'] ?? ''),
      driverId: String(params['driverId'] ?? '')
    }
  }

  consumeLatest(): LaunchIntent {
    return this.latest
  }
}

const STORE: IntentMemoryStore = new IntentMemoryStore()

export function updateLaunchIntent(want: Want): void {
  STORE.updateFromWant(want)
}

export function getLaunchIntent(): LaunchIntent {
  return STORE.consumeLatest()
}
