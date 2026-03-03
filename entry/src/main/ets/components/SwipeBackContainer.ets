import { MotionPresets } from '../common/MotionPresets'
import { NavService } from '../common/NavService'

@Component
export struct SwipeBackContainer {
  @Prop canSwipeBack: boolean = true
  @Prop startInsetVp: number = MotionPresets.swipeStartInset
  @Prop edgeWidthVp: number = MotionPresets.swipeEdgeWidth
  @Prop triggerDistanceVp: number = MotionPresets.swipeTriggerDistance
  @BuilderParam content: () => void = this.defaultContent

  @State private offsetX: number = 0
  @State private contentOpacity: number = 1
  private tracking: boolean = false
  private validSwipe: boolean = false
  private startX: number = 0
  private startY: number = 0

  private resetPosition(): void {
    this.offsetX = 0
    this.contentOpacity = 1
  }

  @Builder
  private defaultContent(): void {
    Column() {}
      .width('100%')
      .height('100%')
  }

  private handleTouch(event: TouchEvent): void {
    if (!this.canSwipeBack) {
      return
    }

    if (event.type === TouchType.Up || event.type === TouchType.Cancel) {
      this.tracking = false
      if (this.offsetX >= this.triggerDistanceVp) {
        this.resetPosition()
        NavService.back()
      } else {
        this.resetPosition()
      }
      return
    }

    if (event.touches.length === 0) {
      return
    }

    const point = event.touches[0]
    if (!point) {
      return
    }

    if (event.type === TouchType.Down) {
      this.startX = point.x
      this.startY = point.y
      this.tracking = true
      this.validSwipe = point.x >= this.startInsetVp && point.x <= this.edgeWidthVp
      return
    }

    if (!this.tracking || !this.validSwipe) {
      return
    }

    if (event.type === TouchType.Move) {
      const dx: number = point.x - this.startX
      const dy: number = Math.abs(point.y - this.startY)
      if (dx <= 0 || dy > dx) {
        return
      }
      this.offsetX = Math.min(120, dx)
      this.contentOpacity = Math.max(0.9, 1 - this.offsetX / 800)
      return
    }
  }

  build() {
    Stack() {
      this.content()
    }
    .width('100%')
    .height('100%')
    .translate({ x: this.offsetX, y: 0 })
    .opacity(this.contentOpacity)
    .animation({ duration: MotionPresets.swipeBackMs })
    .onTouch((event: TouchEvent) => {
      this.handleTouch(event)
    })
  }
}
