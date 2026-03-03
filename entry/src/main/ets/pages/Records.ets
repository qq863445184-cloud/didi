import { GrabRecordItem, GrabRecordState } from '../models/DispatchModels'
import { listGrabRecords } from '../common/GrabRecordStore'
import { NavService } from '../common/NavService'
import { AppColor, FontScale, Radius, ShadowToken, Spacing } from '../common/DesignTokens'
import { MotionPresets } from '../common/MotionPresets'

@Entry
@Component
struct Records {
  @State records: GrabRecordItem[] = []
  @State filter: string = 'all'
  @State pageReady: boolean = false

  aboutToAppear(): void {
    this.records = listGrabRecords()
    this.pageReady = true
  }

  private goBack(): void {
    NavService.back()
  }

  private filtered(): GrabRecordItem[] {
    if (this.filter === 'all') {
      return this.records
    }
    return this.records.filter((item: GrabRecordItem) => item.state === this.filter)
  }

  @Builder
  private buildBackIcon(): void {
    Stack() {
      Circle()
        .width(18)
        .height(18)
        .fill('#E9EEF8')
      Text('‹')
        .fontSize(18)
        .fontColor('#29354F')
        .fontWeight(FontWeight.Medium)
        .margin({ left: -1, top: -1 })
    }
    .width(30)
    .height(30)
    .borderRadius(15)
    .backgroundColor('#F8FAFF')
    .border({ width: 1, color: '#E6ECF7' })
    .onClick(() => {
      this.goBack()
    })
  }

  @Builder
  private buildTopBar(): void {
    Row() {
      this.buildBackIcon()
      Text('抢单清单')
        .fontSize(16)
        .fontWeight(FontWeight.Medium)
        .fontColor(AppColor.textPrimary)
        .margin({ left: 10 })
      Blank()
    }
    .width('100%')
    .height(50)
    .padding({ left: Spacing.pageHorizontal, right: Spacing.pageHorizontal })
    .backgroundColor(AppColor.navGlass)
    .border({ width: { bottom: 1 }, color: AppColor.lineSoft })
  }

  @Builder
  private buildFilterTab(label: string, key: string): void {
    Text(label)
      .fontSize(12)
      .fontWeight(this.filter === key ? FontWeight.Medium : FontWeight.Regular)
      .fontColor(this.filter === key ? '#0F5EE8' : AppColor.textSecondary)
      .padding({ left: 10, right: 10, top: 5, bottom: 5 })
      .backgroundColor(this.filter === key ? '#EAF1FF' : '#F3F6FC')
      .borderRadius(13)
      .animation({ duration: MotionPresets.cardPopMs })
      .onClick(() => {
        this.filter = key
      })
  }

  @Builder
  private buildRecordCard(item: GrabRecordItem): void {
    Column({ space: Spacing.compactRow }) {
      Row() {
        Text(item.summary)
          .fontSize(13)
          .fontWeight(FontWeight.Medium)
          .fontColor(AppColor.textPrimary)
        Blank()
        Text(item.createdAt)
          .fontSize(FontScale.caption)
          .fontColor(AppColor.textTertiary)
      }
      .width('100%')

      Text(item.detail)
        .fontSize(FontScale.body)
        .fontColor(AppColor.textSecondary)
        .maxLines(2)
        .textOverflow({ overflow: TextOverflow.Ellipsis })

      Row() {
        Text(item.state === GrabRecordState.Success ? '成功' : item.state === GrabRecordState.Failed ? '失败' : '进行中')
          .fontSize(FontScale.caption)
          .fontWeight(FontWeight.Medium)
          .fontColor(item.state === GrabRecordState.Success ? '#159251' : item.state === GrabRecordState.Failed ? '#D84D4D' : '#D57B22')
      }
      .width('100%')
    }
    .width('100%')
    .padding(Spacing.cardPadding)
    .backgroundColor('#FFFFFF')
    .borderRadius(Radius.card)
    .border({ width: 1, color: '#EAF0F8' })
    .shadow(ShadowToken.soft)
    .animation({ duration: MotionPresets.cardPopMs })
  }

  build() {
    Column() {
      Column({ space: 8 }) {
        this.buildTopBar()

        Row({ space: 8 }) {
          this.buildFilterTab('全部', 'all')
          this.buildFilterTab('进行中', GrabRecordState.Listening)
          this.buildFilterTab('成功', GrabRecordState.Success)
          this.buildFilterTab('失败', GrabRecordState.Failed)
        }
        .width('100%')
        .padding({ left: Spacing.pageHorizontal, right: Spacing.pageHorizontal, top: 2 })

        List({ space: 8 }) {
          ForEach(this.filtered(), (item: GrabRecordItem) => {
            ListItem() {
              this.buildRecordCard(item)
            }
          }, (item: GrabRecordItem) => item.id)
        }
        .width('100%')
        .layoutWeight(1)
        .padding({ left: Spacing.pageHorizontal, right: Spacing.pageHorizontal, bottom: Spacing.pageHorizontal })
      }
    }
    .width('100%')
    .height('100%')
    .opacity(this.pageReady ? 1 : 0.96)
    .translate({ x: this.pageReady ? 0 : 12, y: 0 })
    .animation({ duration: MotionPresets.pageEnterMs })
    .backgroundColor(AppColor.bgBase)
  }
}
