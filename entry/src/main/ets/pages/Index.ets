import {
  DriverOrderStatus,
  GrabFailResult,
  GrabRecordItem,
  GrabRecordState,
  GrabSuccessResult,
  OrderInfo
} from '../models/DispatchModels'
import { DispatchRepository, HttpDispatchApi } from '../services/DispatchService'
import { addGrabRecord } from '../common/GrabRecordStore'
import { getLaunchIntent } from '../common/IntentStore'
import { NavService } from '../common/NavService'
import { AppColor, FontScale, Radius, ShadowToken, Spacing } from '../common/DesignTokens'
import { MotionPresets } from '../common/MotionPresets'

@Entry
@Component
struct Index {
  private readonly repository: DispatchRepository = new DispatchRepository(new HttpDispatchApi('http://121.43.51.41:5000', ''))
  private activeGrabRequestId: string = ''
  private lastHandledIntentId: string = ''

  @State status: DriverOrderStatus = DriverOrderStatus.Idle
  @State driverId: string = '--'
  @State todayIncome: number = 0
  @State todayTrips: number = 0
  @State score: number = 0
  @State onlineHours: number = 0
  @State messageText: string = '暂无新消息，点击查看历史消息～'
  @State failReason: string = '附近暂无可抢订单，建议切换热区后继续出车'
  @State currentOrder: OrderInfo = {
    pickup: '友谊路/士井一街(路口)东南侧',
    destination: '沙河舒馨家园',
    type: '普通型实时单'
  }

  aboutToAppear(): void {
    this.loadDriverSession().then((driverId) => {
      this.handleLaunchIntent(driverId)
    }).catch(() => {
    })
  }

  private handleLaunchIntent(driverId: string): void {
    const intent = getLaunchIntent()
    if (intent.id !== '' && intent.id !== this.lastHandledIntentId) {
      this.lastHandledIntentId = intent.id
      if (intent.intentType === 'openRecords') {
        this.openRecordPage()
        return
      }
      if (intent.intentType === 'resumeGrab' && intent.requestId !== '') {
        this.status = DriverOrderStatus.Listening
        this.activeGrabRequestId = intent.requestId
        this.messageText = '系统拉起：正在恢复抢单状态'
        this.resolveDispatchResult(intent.requestId)
        return
      }
    }
    this.restorePendingGrabStatus(driverId)
  }

  private startDispatch(): void {
    if (this.status === DriverOrderStatus.Listening) {
      return
    }
    this.status = DriverOrderStatus.Listening
    this.messageText = '系统消息'
    this.failReason = '正在请求服务端派单结果...'
    this.addRecord(GrabRecordState.Listening, '开始抢单', `司机 ${this.driverId} 发起抢单请求`)

    this.repository.startGrab(this.driverId).then((resp) => {
      this.activeGrabRequestId = resp.requestId
      const delay: number = 7000 + Math.floor(Math.random() * 3000)
      setTimeout(() => {
        this.resolveDispatchResult(resp.requestId)
      }, delay)
    }).catch(() => {
      this.applyFailed('启动抢单失败，请检查网络后重试')
    })
  }

  private resolveDispatchResult(requestId: string): void {
    if (this.status !== DriverOrderStatus.Listening || this.activeGrabRequestId !== requestId) {
      return
    }
    this.repository.fetchGrabResult(requestId).then((result) => {
      if (this.activeGrabRequestId !== requestId) {
        return
      }
      if (result.success === true) {
        this.applySuccess(result.order, result.rewardDelta)
      } else {
        const failResult: GrabFailResult = result as GrabFailResult
        this.applyFailed(failResult.reason)
      }
      this.activeGrabRequestId = ''
    }).catch(() => {
      this.applyFailed('获取抢单结果失败，请稍后重试')
      this.activeGrabRequestId = ''
    })
  }

  private applySuccess(order: OrderInfo, rewardDelta: number): void {
    this.currentOrder = order
    this.todayTrips = this.todayTrips + 1
    this.todayIncome = this.todayIncome + rewardDelta
    this.score = this.score + 1
    this.onlineHours = this.onlineHours + 1
    this.status = DriverOrderStatus.Success
    this.addRecord(GrabRecordState.Success, `接单成功：${order.type}`, `${order.pickup} -> ${order.destination}`)
  }

  private applyFailed(reason: string): void {
    this.failReason = reason
    this.status = DriverOrderStatus.Failed
    this.addRecord(GrabRecordState.Failed, '接单失败', reason)
  }

  private backToIdle(): void {
    if (this.activeGrabRequestId !== '') {
      const requestId: string = this.activeGrabRequestId
      this.activeGrabRequestId = ''
      this.repository.stopGrab(requestId).catch(() => {
      })
    }
    this.status = DriverOrderStatus.Idle
    this.messageText = '暂无新消息，点击查看历史消息～'
  }

  private loadDriverSession(): Promise<string> {
    return this.repository.fetchDriverSession().then((session) => {
      this.driverId = session.driverId
      this.todayIncome = session.todayIncome
      this.todayTrips = session.todayTrips
      this.score = session.score
      this.onlineHours = session.onlineHours
      return session.driverId
    }).catch(() => {
      this.messageText = '获取司机会话失败，已使用本地兜底数据'
      return this.driverId
    })
  }

  private restorePendingGrabStatus(driverId: string): void {
    if (driverId === '--') {
      return
    }
    this.repository.fetchPendingGrabStatus(driverId).then((pending) => {
      if (pending.state === 'pending') {
        this.status = DriverOrderStatus.Listening
        this.activeGrabRequestId = pending.requestId
        this.messageText = '检测到系统代抢任务进行中'
        this.addRecord(GrabRecordState.Listening, '系统代抢进行中', `请求ID: ${pending.requestId}`)
        return
      }
      if (pending.state === 'success' && pending.result) {
        const successResult: GrabSuccessResult = pending.result as GrabSuccessResult
        this.applySuccess(successResult.order, successResult.rewardDelta)
        this.repository.acknowledgeSystemResult(pending.requestId).catch(() => {
        })
        return
      }
      if (pending.state === 'failed' && pending.result) {
        const failResult: GrabFailResult = pending.result as GrabFailResult
        this.applyFailed(failResult.reason)
        this.repository.acknowledgeSystemResult(pending.requestId).catch(() => {
        })
      }
    }).catch(() => {
    })
  }

  private onBackTap(): void {
    if (this.status !== DriverOrderStatus.Idle) {
      this.backToIdle()
    }
  }

  private openRecordPage(): void {
    NavService.push('pages/Records')
  }

  private addRecord(state: GrabRecordState, summary: string, detail: string): void {
    const now: Date = new Date()
    const mm: string = `${now.getMonth() + 1}`.padStart(2, '0')
    const dd: string = `${now.getDate()}`.padStart(2, '0')
    const hh: string = `${now.getHours()}`.padStart(2, '0')
    const min: string = `${now.getMinutes()}`.padStart(2, '0')
    const item: GrabRecordItem = {
      id: `${now.getTime()}_${Math.floor(Math.random() * 10000)}`,
      createdAt: `${mm}-${dd} ${hh}:${min}`,
      state,
      summary,
      detail
    }
    addGrabRecord(item)
  }

  @Builder
  private buildBackIcon(): void {
    Stack() {
      Circle()
        .width(18)
        .height(18)
        .fill('#E7EDF8')
      Text('‹')
        .fontSize(18)
        .fontColor('#2A3752')
        .fontWeight(FontWeight.Medium)
        .margin({ left: -1, top: -1 })
    }
    .width(30)
    .height(30)
    .borderRadius(15)
    .backgroundColor('#F8FAFF')
    .border({ width: 1, color: '#E6ECF7' })
    .onClick(() => {
      this.onBackTap()
    })
  }

  @Builder
  private buildAvatarIcon(): void {
    Stack() {
      Circle()
        .width(14)
        .height(14)
        .fill('#E8ECF6')
      Circle()
        .width(5)
        .height(5)
        .fill('#7D8AA8')
        .margin({ top: -3 })
      Column()
        .width(8)
        .height(5)
        .borderRadius(4)
        .backgroundColor('#7D8AA8')
        .margin({ top: 5 })
    }
    .width(20)
    .height(20)
  }

  @Builder
  private buildTabIcon(active: boolean): void {
    Stack() {
      Circle()
        .width(16)
        .height(16)
        .fill(active ? '#FFE3CC' : '#E7EBF2')
      Circle()
        .width(6)
        .height(6)
        .fill(active ? '#FF8A00' : '#9BA6BA')
    }
    .width(16)
    .height(16)
  }

  @Builder
  private buildMessageIcon(): void {
    Stack() {
      Circle()
        .width(13)
        .height(13)
        .fill('#E8EDF7')
      Circle()
        .width(5)
        .height(5)
        .fill('#60729A')
    }
    .width(13)
    .height(13)
  }

  @Builder
  private buildTopNav(): void {
    Row() {
      this.buildBackIcon()
      Blank()
      Text('滴滴模拟接单')
        .fontSize(FontScale.title)
        .fontWeight(FontWeight.Medium)
        .fontColor(AppColor.textPrimary)
      Blank()
      Row() {
        Circle()
          .width(4)
          .height(4)
          .fill('#C1C7D2')
        Circle()
          .width(4)
          .height(4)
          .fill('#C1C7D2')
      }
      .width(20)
      .justifyContent(FlexAlign.SpaceBetween)
    }
    .width('100%')
    .height(44)
    .padding({ left: 12, right: 12 })
    .backgroundColor(AppColor.navGlass)
    .border({ width: { bottom: 1 }, color: AppColor.lineSoft })
  }

  @Builder
  private buildMainTopBar(): void {
    Row() {
      this.buildAvatarIcon()
      Blank()
      Text('首页')
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor('#FFFFFF')
        .padding({ bottom: 6 })
        .border({ width: { bottom: 2 }, color: '#FF7D00' })
      Text('生活')
        .fontSize(14)
        .fontColor('#95A0B8')
        .margin({ left: 22 })
      Blank()
      Text(' ')
        .width(64)
    }
    .width('100%')
    .height(64)
    .padding({ left: 12, right: 12, top: 10 })
    .backgroundColor('#2E3753')
  }

  @Builder
  private buildStatsRow(): void {
    Row() {
      this.buildOneStat(`${this.score}`, '口碑值')
      this.buildOneStat(`${this.todayIncome}`, '今日流水')
      this.buildOneStat(`${this.todayTrips}`, '今日接单')
      this.buildOneStat(`${this.onlineHours}.0`, '在线时长')
      Text('>')
        .fontSize(13)
        .fontColor('#A0A0A0')
    }
    .width('100%')
    .height(56)
    .padding({ left: 14, right: 12 })
    .backgroundColor('#FFFFFF')
    .border({ width: { bottom: 1 }, color: AppColor.lineSoft })
  }

  @Builder
  private buildOneStat(v: string, t: string): void {
    Column({ space: 2 }) {
      Text(v)
        .fontSize(17)
        .fontWeight(FontWeight.Medium)
        .fontColor('#2C2C2C')
      Text(t)
        .fontSize(FontScale.caption)
        .fontColor('#5B5B5B')
    }
    .width('24%')
  }

  @Builder
  private buildTabRow(): void {
    Row() {
      this.buildTabCell('选单', false)
      this.buildTabCell('奖励', false)
      this.buildTabCell('热力图', true)
      this.buildTabCell('清单', false, () => {
        this.openRecordPage()
      })
    }
    .width('100%')
    .height(42)
    .padding({ left: 10, right: 10 })
    .backgroundColor('#FFFFFF')
    .border({ width: { bottom: 1 }, color: AppColor.lineSoft })
  }

  @Builder
  private buildTabCell(name: string, active: boolean, onTap?: () => void): void {
    Column({ space: 3 }) {
      this.buildTabIcon(active)
      Text(name)
        .fontSize(12)
        .fontWeight(active ? FontWeight.Medium : FontWeight.Regular)
        .fontColor(active ? '#1F1F1F' : '#555B66')
    }
    .alignItems(HorizontalAlign.Center)
    .justifyContent(FlexAlign.Center)
    .width('25%')
    .height(30)
    .borderRadius(6)
    .backgroundColor(active ? '#F5F6F8' : '#FFFFFF')
    .border(active ? { width: 1, color: '#E9ECF2' } : { width: 0 })
    .onClick(() => {
      if (onTap) {
        onTap()
      }
    })
  }

  @Builder
  private buildMessageRow(): void {
    Row() {
      this.buildMessageIcon()
      Text(` ${this.messageText}`)
        .fontSize(FontScale.body)
        .fontColor('#4D4D4D')
        .maxLines(1)
      Blank()
      if (this.status === DriverOrderStatus.Listening) {
        Text('16:01')
          .fontSize(10)
          .fontColor('#A2A2A2')
          .margin({ right: 6 })
      }
      Text('>')
        .fontSize(12)
        .fontColor('#B0B0B0')
    }
    .width('100%')
    .height(36)
    .padding({ left: 12, right: 12 })
    .backgroundColor('#FFFFFF')
  }

  @Builder
  private buildMapPlaceholder(): void {
    Stack() {
      Column() {}
      .width('100%')
      .height('100%')
      .backgroundColor('#E3E8F1')

      // map land blocks
      Column()
        .width(132)
        .height(84)
        .backgroundColor('#D0D8E8')
        .borderRadius(16)
        .margin({ left: 26, top: 118 })
        .align(Alignment.TopStart)
        .rotate({ angle: -8 })
      Column()
        .width(118)
        .height(86)
        .backgroundColor('#CDD6E7')
        .borderRadius(16)
        .margin({ left: 230, top: 132 })
        .align(Alignment.TopStart)
        .rotate({ angle: 7 })
      Column()
        .width(122)
        .height(96)
        .backgroundColor('#CAD4E5')
        .borderRadius(16)
        .margin({ left: 188, top: 310 })
        .align(Alignment.TopStart)
        .rotate({ angle: -6 })
      Column()
        .width(96)
        .height(72)
        .backgroundColor('#D4DDEB')
        .borderRadius(14)
        .margin({ left: 78, top: 214 })
        .align(Alignment.TopStart)
        .rotate({ angle: 8 })

      // primary roads
      Column()
        .width(430)
        .height(14)
        .backgroundColor('#F8FAFD')
        .borderRadius(10)
        .margin({ left: -18, top: 248 })
        .align(Alignment.TopStart)
      Column()
        .width(406)
        .height(14)
        .backgroundColor('#F8FAFD')
        .borderRadius(10)
        .margin({ left: 8, top: 286 })
        .align(Alignment.TopStart)
        .rotate({ angle: -7 })
      Column()
        .width(388)
        .height(14)
        .backgroundColor('#F8FAFD')
        .borderRadius(10)
        .margin({ left: 6, top: 334 })
        .align(Alignment.TopStart)
        .rotate({ angle: 6 })

      // secondary roads
      Column()
        .width(400)
        .height(6)
        .backgroundColor('#DCE3EE')
        .borderRadius(6)
        .margin({ left: -6, top: 230 })
        .align(Alignment.TopStart)
        .rotate({ angle: 9 })
      Column()
        .width(388)
        .height(6)
        .backgroundColor('#DCE3EE')
        .borderRadius(6)
        .margin({ left: 18, top: 364 })
        .align(Alignment.TopStart)
        .rotate({ angle: -9 })
      Column()
        .width(244)
        .height(6)
        .backgroundColor('#DCE3EE')
        .borderRadius(6)
        .margin({ left: 102, top: 176 })
        .align(Alignment.TopStart)
        .rotate({ angle: 14 })
      Column()
        .width(6)
        .height(168)
        .backgroundColor('#DCE3EE')
        .borderRadius(6)
        .margin({ left: 146, top: 116 })
        .align(Alignment.TopStart)
        .rotate({ angle: -12 })
      Column()
        .width(6)
        .height(138)
        .backgroundColor('#DCE3EE')
        .borderRadius(6)
        .margin({ left: 278, top: 196 })
        .align(Alignment.TopStart)
        .rotate({ angle: 10 })

      // soft area overlay for map depth
      Column()
        .width(220)
        .height(120)
        .backgroundColor('#E7EDF7')
        .borderRadius(24)
        .margin({ left: 116, top: 168 })
        .align(Alignment.TopStart)
        .rotate({ angle: -4 })

      // simplified connected route: straight line from start to end
      Column()
        .width(250)
        .height(14)
        .backgroundColor('#76D9BA')
        .borderRadius(12)
        .margin({ left: 62, top: 250 })
        .align(Alignment.TopStart)
      Column()
        .width(250)
        .height(12)
        .backgroundColor('#19C07F')
        .borderRadius(12)
        .margin({ left: 62, top: 251 })
        .align(Alignment.TopStart)

      Row() {
        Text('起')
          .fontSize(12)
          .fontColor('#FFFFFF')
      }
      .width(26)
      .height(26)
      .justifyContent(FlexAlign.Center)
      .borderRadius(13)
      .backgroundColor('#26B680')
      .align(Alignment.TopStart)
      .margin({ left: 56, top: 238 })
      .shadow({ radius: 6, color: '#30000000', offsetX: 0, offsetY: 1 })

      Row() {
        Text('终')
          .fontSize(12)
          .fontColor('#FFFFFF')
      }
      .width(26)
      .height(26)
      .justifyContent(FlexAlign.Center)
      .borderRadius(13)
      .backgroundColor('#2FA2FF')
      .align(Alignment.TopStart)
      .margin({ left: 296, top: 238 })
      .shadow({ radius: 6, color: '#30000000', offsetX: 0, offsetY: 1 })

      Circle()
        .width(36)
        .height(36)
        .fill('#3FAEFF33')
        .align(Alignment.TopStart)
        .margin({ left: 291, top: 233 })

      // lightweight map controls (right side)
      Column({ space: 6 }) {
        Row() {
          Text('⌖')
            .fontSize(11)
            .fontColor('#6E87B8')
        }
        .width(26)
        .height(26)
        .justifyContent(FlexAlign.Center)
        .borderRadius(13)
        .backgroundColor('#F7F9FC')

        Row() {
          Text('层')
            .fontSize(9)
            .fontColor('#6B7A96')
        }
        .width(26)
        .height(26)
        .justifyContent(FlexAlign.Center)
        .borderRadius(13)
        .backgroundColor('#F7F9FC')
      }
      .padding({ left: 4, right: 4, top: 4, bottom: 4 })
      .backgroundColor('#EEF2F899')
      .borderRadius(15)
      .align(Alignment.TopEnd)
      .margin({ right: 8, top: 128 })
      .shadow({ radius: 4, color: '#1E000000', offsetX: 0, offsetY: 1 })

    }
    .layoutWeight(1)
    .width('100%')
  }

  @Builder
  private buildShieldIcon(): void {
    Stack() {
      Circle()
        .width(40)
        .height(40)
        .fill('#78A9FF')
      Column()
        .width(16)
        .height(20)
        .backgroundColor('#FFFFFF')
        .borderRadius({ topLeft: 8, topRight: 8, bottomLeft: 10, bottomRight: 10 })
      Text('✓')
        .fontSize(10)
        .fontWeight(FontWeight.Bold)
        .fontColor('#4A86F0')
        .margin({ top: -1 })
    }
    .width(40)
    .height(40)
  }

  @Builder
  private buildCalendarIcon(): void {
    Stack() {
      Circle()
        .width(40)
        .height(40)
        .fill('#FF9A57')
      Column() {
        Row() {}
          .width(16)
          .height(4)
          .backgroundColor('#FFFFFF')
          .borderRadius(2)
        Column({ space: 2 }) {
          Row({ space: 2 }) {
            Circle().width(2).height(2).fill('#FFFFFF')
            Circle().width(2).height(2).fill('#FFFFFF')
            Circle().width(2).height(2).fill('#FFFFFF')
          }
          Row({ space: 2 }) {
            Circle().width(2).height(2).fill('#FFFFFF')
            Circle().width(2).height(2).fill('#FFFFFF')
            Circle().width(2).height(2).fill('#FFFFFF')
          }
        }
        .margin({ top: 3 })
      }
      .width(16)
      .alignItems(HorizontalAlign.Center)
    }
    .width(40)
    .height(40)
  }

  @Builder
  private buildFeedCard(title: string, desc: string, iconType: string, tagText: string): void {
    Column({ space: 5 }) {
      Row() {
        Text(tagText)
          .fontSize(FontScale.caption)
          .fontColor('#FFFFFF')
          .padding({ left: 7, right: 7, top: 2, bottom: 2 })
          .backgroundColor('#5E7FB8')
          .borderRadius(6)
        Blank()
        Text('×')
          .fontSize(14)
          .fontColor('#A9AFBA')
      }
      .width('100%')
      .height(18)
      .alignItems(VerticalAlign.Center)

      Row() {
        if (iconType === 'shield') {
          this.buildShieldIcon()
        } else {
          this.buildCalendarIcon()
        }

        Column({ space: 5 }) {
          Text(title)
            .fontSize(13)
            .fontWeight(FontWeight.Medium)
            .fontColor('#1F1F1F')
          Text(desc)
            .fontSize(FontScale.caption)
            .fontColor('#656565')
            .maxLines(2)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }
        .margin({ left: 10 })
        Blank()
        Text('>')
          .fontSize(12)
          .fontColor('#B5BBC6')
      }
      .width('100%')
      .alignItems(VerticalAlign.Center)
    }
    .width('100%')
    .height(82)
    .padding({ left: 10, right: 10, top: 7, bottom: 7 })
    .backgroundColor('#FFFFFF')
    .borderRadius(Radius.card)
    .border({ width: 1, color: '#E9EEF7' })
    .shadow(ShadowToken.soft)
    .animation({ duration: MotionPresets.cardPopMs })
  }

  @Builder
  private buildMiddleFloatIcon(): void {
    Row() {
      Text('🛡')
        .fontSize(16)
    }
    .width(40)
    .height(40)
    .justifyContent(FlexAlign.Center)
    .backgroundColor('#FFFFFF')
    .borderRadius(10)
    .shadow({ radius: 8, color: '#22000000', offsetX: 0, offsetY: 2 })
  }

  @Builder
  private buildBottomPanel(): void {
    Column({ space: 8 }) {
      if (this.status === DriverOrderStatus.Failed) {
        Row() {
          Text('未匹配成功')
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor('#E65050')
          Blank()
          Text('模式')
            .fontSize(12)
            .fontColor('#606060')
        }
        .width('100%')
        Text(this.failReason)
          .fontSize(12)
          .fontColor('#333333')
      } else if (this.status === DriverOrderStatus.Success) {
        Row({ space: 6 }) {
          Text('实时')
            .fontSize(10)
            .fontColor('#DDE4F2')
            .padding({ left: 6, right: 6, top: 2, bottom: 2 })
            .backgroundColor('#4A5572')
            .borderRadius(5)
          Text('极速应答')
            .fontSize(10)
            .fontColor('#DDE4F2')
            .padding({ left: 6, right: 6, top: 2, bottom: 2 })
            .backgroundColor('#4A5572')
            .borderRadius(5)
        }
        Text(this.currentOrder.type)
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor('#FFFFFF')
        Text(`• ${this.currentOrder.pickup}`)
          .fontSize(12)
          .fontColor('#DDE4F2')
        Text(`• ${this.currentOrder.destination}`)
          .fontSize(12)
          .fontColor('#DDE4F2')
        Text('乘客很着急')
          .fontSize(11)
          .fontColor('#FFAE6A')
          .padding({ left: 8, right: 8, top: 4, bottom: 4 })
          .backgroundColor('#4A5572')
          .borderRadius(7)
      } else {
        Row() {
          Text(this.status === DriverOrderStatus.Listening ? '指派模式正在持续为您匹配订单' : '勾选多类订单后点击出车，更快有单！')
            .fontSize(FontScale.body)
            .fontColor('#2B2B2B')
          Blank()
          Text('模式')
            .fontSize(12)
            .fontColor('#606060')
        }
        .width('100%')
      }

      if (this.status === DriverOrderStatus.Listening) {
        Row() {
          Text('✓ 正常抢单')
            .fontSize(FontScale.caption)
            .fontColor('#FFFFFF')
            .fontWeight(FontWeight.Medium)
        }
        .padding({ left: 10, right: 10, top: 4, bottom: 4 })
        .backgroundColor('#57B560')
        .borderRadius(10)

        Row() {
          Text('《 听单中 》')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor('#52A55C')
          Blank()
          Text('收车')
            .fontSize(14)
            .fontColor('#1F1F1F')
        }
        .width('100%')
        .height(44)
        .padding({ left: 14, right: 14 })
        .backgroundColor('#F3F5F7')
        .borderRadius(22)
        .onClick(() => {
          this.backToIdle()
        })
      } else if (this.status === DriverOrderStatus.Success) {
        Text('已为你锁定乘客，请立即前往上车点')
          .fontSize(11)
          .fontColor('#B8C4DA')
      } else {
        Button(this.status === DriverOrderStatus.Failed ? '重新出车' : '出车')
          .width('100%')
          .height(44)
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .fontColor('#FFFFFF')
          .backgroundColor(AppColor.accent)
          .borderRadius(Radius.pill)
          .animation({ duration: MotionPresets.pressMs })
          .onClick(() => {
            this.startDispatch()
          })
      }
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start)
    .padding({ left: Spacing.pageHorizontal, right: Spacing.pageHorizontal, top: 12, bottom: 14 })
    .backgroundColor(this.status === DriverOrderStatus.Success ? '#2F3D58' : '#FFFFFF')
    .borderRadius({ topLeft: Radius.panel, topRight: Radius.panel })
    .shadow(this.status === DriverOrderStatus.Success ? ShadowToken.panel : ShadowToken.none)
    .animation({ duration: MotionPresets.panelLiftMs })
  }

  build() {
    Column() {
      Stack() {
        Column()
          .width('100%')
          .height('100%')
          .backgroundColor(AppColor.bgBase)

        Column({ space: 0 }) {
          if (this.status === DriverOrderStatus.Success) {
            this.buildMapPlaceholder()
            Row() {
              Column()
                .width(36)
                .height(3)
                .borderRadius(3)
                .backgroundColor('#CCD3E0')
            }
            .width('100%')
            .height(10)
            .justifyContent(FlexAlign.Center)
            .backgroundColor('#E7ECF5')
          } else {
            this.buildMainTopBar()
            this.buildStatsRow()
            this.buildTabRow()
            this.buildMessageRow()

            if (this.status === DriverOrderStatus.Listening) {
              Column({ space: 8 }) {
                this.buildFeedCard('已接入滴滴安全中心', '为维护司乘安全，平台已自动开启全程录音保护', 'shield', '系统消息')
                this.buildFeedCard('快来查看今天的车主日报', '收到一份 08 日车主日报，请赶快查收吧', 'calendar', '车主日报')
              }
              .width('100%')
              .padding({ left: Spacing.pageHorizontal, right: Spacing.pageHorizontal, top: Spacing.pageHorizontal })
              .layoutWeight(1)
              .backgroundColor('#F4F7FC')
            } else {
              Stack() {
                Column()
                  .width('100%')
                  .height('100%')
                  .backgroundColor('#F4F7FC')
                Row() {
                  this.buildMiddleFloatIcon()
                }
                .align(Alignment.BottomStart)
                .margin({ left: Spacing.pageHorizontal, bottom: 18 })
              }
              .width('100%')
              .layoutWeight(1)
            }
          }

          this.buildBottomPanel()
        }
        .width('100%')
        .height('100%')
        .opacity(1)
        .animation({ duration: MotionPresets.pageEnterMs })
      }
      .width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
  }
}
