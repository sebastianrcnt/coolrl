const MAX_IOS_CANVAS_DPR = 1.5;
const MAX_MOBILE_CANVAS_DPR = 2;

interface WindowLike {
  matchMedia?: (query: string) => MediaQueryList;
  innerWidth: number;
  devicePixelRatio?: number;
}

interface NavigatorLike {
  userAgent: string;
  platform: string;
  maxTouchPoints: number;
}

export interface DeviceEnvironment {
  readonly isMobile: boolean;
  readonly isIos: boolean;
  readonly isWebKit: boolean;
  readonly canvasPixelRatio: number;
  readonly isLowMemoryMode: boolean;
}

export function isMobileDevice(win: WindowLike = window): boolean {
  const match = win.matchMedia ? win.matchMedia.bind(win) : null;
  const narrow = match ? match("(max-width: 719px)").matches : win.innerWidth < 720;
  const coarse = match ? match("(pointer: coarse)").matches : false;
  return narrow || coarse;
}

export function isIosDevice(nav: NavigatorLike = navigator): boolean {
  const ua = nav.userAgent || "";
  const platform = nav.platform || "";
  return (
    /iPad|iPhone|iPod/.test(ua) ||
    (platform === "MacIntel" && nav.maxTouchPoints > 1)
  );
}

export function isWebKitBrowser(nav: NavigatorLike = navigator): boolean {
  const ua = nav.userAgent || "";
  if (isIosDevice(nav)) return true;
  return (
    /AppleWebKit/.test(ua) &&
    /Safari/.test(ua) &&
    !/Chrome|Chromium|CriOS|FxiOS|Edg|OPR|SamsungBrowser|Android/.test(ua)
  );
}

export function canvasPixelRatio(
  win: WindowLike = window,
  nav: NavigatorLike = navigator
): number {
  const dpr = win.devicePixelRatio || 1;
  if (isIosDevice(nav)) return Math.min(dpr, MAX_IOS_CANVAS_DPR);
  return isMobileDevice(win) ? Math.min(dpr, MAX_MOBILE_CANVAS_DPR) : dpr;
}

export function readDeviceEnvironment(
  win: WindowLike = window,
  nav: NavigatorLike = navigator
): DeviceEnvironment {
  const mobile = isMobileDevice(win);
  const ios = isIosDevice(nav);
  return {
    isMobile: mobile,
    isIos: ios,
    isWebKit: isWebKitBrowser(nav),
    canvasPixelRatio: canvasPixelRatio(win, nav),
    isLowMemoryMode: mobile,
  };
}
