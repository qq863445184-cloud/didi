import { router } from '@kit.ArkUI'

export class NavService {
  static push(url: string): void {
    router.pushUrl({ url })
  }

  static back(): void {
    try {
      const stackLen: number = Number(router.getLength())
      if (stackLen > 1) {
        router.back()
        return
      }
      router.replaceUrl({ url: 'pages/Index' })
    } catch (err) {
      router.replaceUrl({ url: 'pages/Index' })
    }
  }
}
