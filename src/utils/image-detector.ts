export interface ImagePayload {
  mime: string;
  bytes: Uint8Array;
}

/**
 * Detect if the given data contains an image payload.
 * Supports PNG, JPEG, GIF, and WebP formats.
 * Returns the image MIME type and the image bytes, or null if no image is detected.
 */
export function extractImagePayload(data: Uint8Array): ImagePayload | null {
  if (data.length < 4) return null;

  const checkHeader = (offset: number, magic: number[]) => {
    if (offset + magic.length > data.length) return false;
    for (let i = 0; i < magic.length; i++) {
      if (data[offset + i] !== magic[i]) return false;
    }
    return true;
  };

  const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  const JPEG_MAGIC = [0xff, 0xd8, 0xff];

  if (checkHeader(0, PNG_MAGIC)) {
    return { mime: 'image/png', bytes: data };
  }

  if (checkHeader(0, JPEG_MAGIC)) {
    return { mime: 'image/jpeg', bytes: data };
  }

  const GIF_MAGIC = [0x47, 0x49, 0x46, 0x38];
  const WEBP_MAGIC = [0x52, 0x49, 0x46, 0x46];

  for (let offset = 0; offset <= Math.min(16, data.length - 4); offset++) {
    if (checkHeader(offset, GIF_MAGIC) && data.length > offset + 4) {
      const nextByte = data[offset + 4];
      if (nextByte === 0x37 || nextByte === 0x39) {
        return { mime: 'image/gif', bytes: data.slice(offset) };
      }
    }
    if (checkHeader(offset, WEBP_MAGIC) && data.length > offset + 8) {
      if (data[offset + 8] === 0x57 && data[offset + 9] === 0x45 && data[offset + 10] === 0x42 && data[offset + 11] === 0x50) {
        return { mime: 'image/webp', bytes: data.slice(offset) };
      }
    }
  }

  return null;
}
