import {
  AutoGrabTaskPayload,
  AutoGrabTaskResponse,
  DriverSession,
  GrabResult,
  GrabStartResponse,
  OrderInfo,
  PendingGrabState,
  PendingGrabStatus
} from '../models/DispatchModels'
import { http } from '@kit.NetworkKit'

interface RideGrabRequest {
  order_id: string
  city: string
  driver_id: string
}

interface ApiResponse {
  httpCode?: number
  success?: boolean
  code?: number
  status?: string
  message?: string
  msg?: string
  error?: string
  data?: Object
}

interface HttpRequestHeaders {
  'Content-Type': string
  Authorization?: string
}

export interface DispatchApi {
  fetchDriverSession(): Promise<DriverSession>
  startGrab(driverId: string): Promise<GrabStartResponse>
  fetchGrabResult(requestId: string): Promise<GrabResult>
  stopGrab(requestId: string): Promise<void>
  submitSystemAutoGrabTask(payload: AutoGrabTaskPayload): Promise<AutoGrabTaskResponse>
  fetchPendingGrabStatus(driverId: string): Promise<PendingGrabStatus>
  acknowledgeSystemResult(requestId: string): Promise<void>
}

export class MockDispatchApi implements DispatchApi {
  private readonly mockPendingByDriver: Map<string, PendingGrabStatus> = new Map<string, PendingGrabStatus>()

  private readonly successOrders: OrderInfo[] = [
    {
      pickup: '友谊路/士井一街(路口)东南侧',
      destination: '沙河舒馨家园',
      type: '普通型实时单'
    },
    {
      pickup: '龙阳路地铁站 3 号口',
      destination: '浦东国际机场 T2 航站楼',
      type: '优享型实时单'
    },
    {
      pickup: '世纪大道地铁站 9 号口',
      destination: '虹桥火车站',
      type: '普通型预约单'
    }
  ]

  private readonly failReasons: string[] = [
    '附近暂无可抢订单，建议切换热区后继续出车',
    '该订单已被其他司机接走，请继续听单',
    '乘客取消订单，系统将继续为你匹配'
  ]

  async fetchDriverSession(): Promise<DriverSession> {
    return {
      driverId: 'DRV-9527',
      todayIncome: 0,
      todayTrips: 0,
      score: 0,
      onlineHours: 0
    }
  }

  async startGrab(driverId: string): Promise<GrabStartResponse> {
    const now: string = Date.now().toString()
    const requestId: string = `${driverId}_${now}`
    this.mockPendingByDriver.set(driverId, {
      requestId,
      state: PendingGrabState.Pending
    })
    return { requestId }
  }

  async fetchGrabResult(requestId: string): Promise<GrabResult> {
    const chance: number = Math.random()
    if (chance < 0.15) {
      const failResult: GrabResult = {
        success: false,
        reason: this.failReasons[Math.floor(Math.random() * this.failReasons.length)]
      }
      return failResult
    }

    const order: OrderInfo = this.successOrders[Math.floor(Math.random() * this.successOrders.length)]
    const successResult: GrabResult = {
      success: true,
      order,
      rewardDelta: 35 + Math.floor(Math.random() * 40)
    }
    return successResult
  }

  async stopGrab(requestId: string): Promise<void> {
    return
  }

  async submitSystemAutoGrabTask(payload: AutoGrabTaskPayload): Promise<AutoGrabTaskResponse> {
    const now: string = Date.now().toString()
    const taskId: string = `TASK_${payload.driverId}_${now}`
    this.mockPendingByDriver.set(payload.driverId, {
      requestId: taskId,
      state: PendingGrabState.Pending
    })
    return { taskId }
  }

  async fetchPendingGrabStatus(driverId: string): Promise<PendingGrabStatus> {
    const pending: PendingGrabStatus | undefined = this.mockPendingByDriver.get(driverId)
    if (!pending) {
      return { requestId: '', state: PendingGrabState.None }
    }
    if (pending.state === PendingGrabState.Pending) {
      const result: GrabResult = await this.fetchGrabResult(pending.requestId)
      if (result.success) {
        const done: PendingGrabStatus = {
          requestId: pending.requestId,
          state: PendingGrabState.Success,
          result
        }
        this.mockPendingByDriver.set(driverId, done)
        return done
      }
      const failed: PendingGrabStatus = {
        requestId: pending.requestId,
        state: PendingGrabState.Failed,
        result
      }
      this.mockPendingByDriver.set(driverId, failed)
      return failed
    }
    return pending
  }

  async acknowledgeSystemResult(requestId: string): Promise<void> {
    this.mockPendingByDriver.forEach((value, key) => {
      if (value.requestId === requestId) {
        this.mockPendingByDriver.delete(key)
      }
    })
    return
  }
}

export class HttpDispatchApi implements DispatchApi {
  private readonly baseUrl: string
  private readonly token: string
  private readonly resultByRequestId: Map<string, GrabResult> = new Map<string, GrabResult>()
  private readonly pendingByDriverId: Map<string, string> = new Map<string, string>()

  constructor(baseUrl: string, token: string) {
    this.baseUrl = baseUrl
    this.token = token
  }

  async fetchDriverSession(): Promise<DriverSession> {
    return {
      driverId: 'DRV-HTTP-001',
      todayIncome: 0,
      todayTrips: 0,
      score: 0,
      onlineHours: 0
    }
  }

  async startGrab(driverId: string): Promise<GrabStartResponse> {
    const requestId: string = `${driverId}_${Date.now()}`
    const body: RideGrabRequest = {
      order_id: requestId,
      city: 'Hangzhou',
      driver_id: driverId
    }

    const data: ApiResponse = await this.postJson('/api/ride-hailing/grab', body)
    const isSuccess: boolean = this.isApiSuccess(data)

    if (isSuccess) {
      const result: GrabResult = {
        success: true,
        order: {
          pickup: '友谊路/士井一街(路口)东南侧',
          destination: '沙河舒馨家园',
          type: '普通型实时单'
        },
        rewardDelta: 42
      }
      this.resultByRequestId.set(requestId, result)
    } else {
      const reason: string = this.extractMessage(data, '服务端返回失败')
      this.resultByRequestId.set(requestId, {
        success: false,
        reason
      })
    }

    this.pendingByDriverId.set(driverId, requestId)
    return { requestId }
  }

  async fetchGrabResult(requestId: string): Promise<GrabResult> {
    const result: GrabResult | undefined = this.resultByRequestId.get(requestId)
    if (!result) {
      return {
        success: false,
        reason: '暂无抢单结果，请稍后重试'
      }
    }
    return result
  }

  async stopGrab(requestId: string): Promise<void> {
    this.resultByRequestId.delete(requestId)
    this.pendingByDriverId.forEach((id, key) => {
      if (id === requestId) {
        this.pendingByDriverId.delete(key)
      }
    })
    return
  }

  async submitSystemAutoGrabTask(payload: AutoGrabTaskPayload): Promise<AutoGrabTaskResponse> {
    const resp: GrabStartResponse = await this.startGrab(payload.driverId)
    return { taskId: resp.requestId }
  }

  async fetchPendingGrabStatus(driverId: string): Promise<PendingGrabStatus> {
    const requestId: string | undefined = this.pendingByDriverId.get(driverId)
    if (!requestId) {
      return { requestId: '', state: PendingGrabState.None }
    }
    const result: GrabResult | undefined = this.resultByRequestId.get(requestId)
    if (!result) {
      return { requestId, state: PendingGrabState.Pending }
    }
    if (result.success) {
      return { requestId, state: PendingGrabState.Success, result }
    }
    return { requestId, state: PendingGrabState.Failed, result }
  }

  async acknowledgeSystemResult(requestId: string): Promise<void> {
    await this.stopGrab(requestId)
    return
  }

  private async postJson(path: string, payload: RideGrabRequest): Promise<ApiResponse> {
    const url: string = `${this.baseUrl}${path}`
    let headers: HttpRequestHeaders = { 'Content-Type': 'application/json' }
    if (this.token !== '') {
      headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.token}`
      }
    }

    const httpRequest = http.createHttp()
    const response = await httpRequest.request(url, {
      method: http.RequestMethod.POST,
      header: headers,
      extraData: JSON.stringify(payload),
      connectTimeout: 15000,
      readTimeout: 15000,
      expectDataType: http.HttpDataType.STRING
    })
    const text: string = String(response.result ?? '')
    const statusCode: number = Number(response.responseCode ?? 0)
    let data: ApiResponse = {}
    try {
      data = JSON.parse(text) as ApiResponse
    } catch (err) {
      httpRequest.destroy()
      throw new Error(`接口返回非JSON: ${text}`)
    }
    data.httpCode = statusCode
    httpRequest.destroy()
    return data
  }

  private isApiSuccess(data: ApiResponse): boolean {
    if (data.success === false) {
      return false
    }
    if (data.success === true) {
      return true
    }
    if (typeof data.code === 'number' && Number(data.code) === 0) {
      return true
    }
    if (typeof data.status === 'string' && String(data.status).toLowerCase() === 'ok') {
      return true
    }
    if (typeof data.httpCode === 'number' && data.httpCode >= 200 && data.httpCode < 300) {
      return true
    }
    return false
  }

  private extractMessage(data: ApiResponse, fallback: string): string {
    if (typeof data.message === 'string') {
      return String(data.message)
    }
    if (typeof data.msg === 'string') {
      return String(data.msg)
    }
    if (typeof data.error === 'string') {
      return String(data.error)
    }
    return fallback
  }
}

export class DispatchRepository {
  private readonly api: DispatchApi

  constructor(api: DispatchApi) {
    this.api = api
  }

  fetchDriverSession(): Promise<DriverSession> {
    return this.api.fetchDriverSession()
  }

  startGrab(driverId: string): Promise<GrabStartResponse> {
    return this.api.startGrab(driverId)
  }

  fetchGrabResult(requestId: string): Promise<GrabResult> {
    return this.api.fetchGrabResult(requestId)
  }

  stopGrab(requestId: string): Promise<void> {
    return this.api.stopGrab(requestId)
  }

  submitSystemAutoGrabTask(payload: AutoGrabTaskPayload): Promise<AutoGrabTaskResponse> {
    return this.api.submitSystemAutoGrabTask(payload)
  }

  fetchPendingGrabStatus(driverId: string): Promise<PendingGrabStatus> {
    return this.api.fetchPendingGrabStatus(driverId)
  }

  acknowledgeSystemResult(requestId: string): Promise<void> {
    return this.api.acknowledgeSystemResult(requestId)
  }
}
