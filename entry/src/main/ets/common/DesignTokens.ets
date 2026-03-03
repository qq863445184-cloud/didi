export interface AppShadowStyle {
  radius: number
  color: string
  offsetX: number
  offsetY: number
}

export class Spacing {
  static readonly pageHorizontal: number = 12
  static readonly sectionGap: number = 8
  static readonly cardPadding: number = 10
  static readonly compactRow: number = 6
}

export class Radius {
  static readonly card: number = 10
  static readonly panel: number = 18
  static readonly pill: number = 20
  static readonly icon: number = 12
}

export class AppColor {
  static readonly bgBase: string = '#EEF2F8'
  static readonly bgElevated: string = '#F7F9FC'
  static readonly navGlass: string = '#F5F8FFEE'
  static readonly textPrimary: string = '#1E2430'
  static readonly textSecondary: string = '#5D6A84'
  static readonly textTertiary: string = '#8C97AB'
  static readonly lineSoft: string = '#E7ECF4'
  static readonly accent: string = '#FF7C28'
}

export class ShadowToken {
  static readonly none: AppShadowStyle = {
    radius: 0,
    color: '#00000000',
    offsetX: 0,
    offsetY: 0
  }

  static readonly soft: AppShadowStyle = {
    radius: 8,
    color: '#1A000000',
    offsetX: 0,
    offsetY: 2
  }

  static readonly panel: AppShadowStyle = {
    radius: 12,
    color: '#24000000',
    offsetX: 0,
    offsetY: -2
  }
}

export class FontScale {
  static readonly title: number = 15
  static readonly body: number = 12
  static readonly caption: number = 10
}
