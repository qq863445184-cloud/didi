import { GrabRecordItem } from '../models/DispatchModels'

class GrabRecordMemoryStore {
  private items: GrabRecordItem[] = []

  add(item: GrabRecordItem): void {
    this.items.unshift(item)
    if (this.items.length > 80) {
      this.items = this.items.slice(0, 80)
    }
  }

  list(): GrabRecordItem[] {
    return this.items
  }
}

const STORE: GrabRecordMemoryStore = new GrabRecordMemoryStore()

export function addGrabRecord(item: GrabRecordItem): void {
  STORE.add(item)
}

export function listGrabRecords(): GrabRecordItem[] {
  return STORE.list()
}
