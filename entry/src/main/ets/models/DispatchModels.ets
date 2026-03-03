export enum DriverOrderStatus {
  Idle = 'idle',
  Listening = 'listening',
  Success = 'success',
  Failed = 'failed'
}

export interface OrderInfo {
  pickup: string
  destination: string
  type: string
}

export interface DriverSession {
  driverId: string
  todayIncome: number
  todayTrips: number
  score: number
  onlineHours: number
}

export interface GrabStartResponse {
  requestId: string
}

export interface GrabFailResult {
  success: false
  reason: string
}

export interface GrabSuccessResult {
  success: true
  order: OrderInfo
  rewardDelta: number
}

export type GrabResult = GrabFailResult | GrabSuccessResult

export enum PendingGrabState {
  None = 'none',
  Pending = 'pending',
  Success = 'success',
  Failed = 'failed'
}

export interface AutoGrabTaskPayload {
  driverId: string
  grabApi: string
  method: string
  token: string
  extraHeaders?: string
  body?: string
}

export interface AutoGrabTaskResponse {
  taskId: string
}

export interface PendingGrabStatus {
  requestId: string
  state: PendingGrabState
  result?: GrabResult
}

export enum GrabRecordState {
  Listening = 'listening',
  Success = 'success',
  Failed = 'failed'
}

export interface GrabRecordItem {
  id: string
  createdAt: string
  state: GrabRecordState
  summary: string
  detail: string
}
