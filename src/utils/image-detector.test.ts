import { describe, it, expect } from 'vitest';
import { extractImagePayload } from './image-detector';

describe('extractImagePayload', () => {
  describe('PNG detection', () => {
    it('should detect PNG image', () => {
      // PNG signature: 89 50 4E 47 0D 0A 1A 0A
      const pngData = new Uint8Array([
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG signature
        0x00, 0x00, 0x00, 0x0d, // IHDR chunk length
        0x49, 0x48, 0x44, 0x52, // IHDR chunk type
        // ... rest of PNG data
      ]);

      const result = extractImagePayload(pngData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/png');
      expect(result?.bytes).toStrictEqual(pngData);
    });

    it('should not detect non-PNG data', () => {
      const notPng = new Uint8Array([0x00, 0x01, 0x02, 0x03]);
      const result = extractImagePayload(notPng);
      expect(result).toBeNull();
    });
  });

  describe('JPEG detection', () => {
    it('should detect JPEG image', () => {
      // JPEG signature: FF D8 FF
      const jpegData = new Uint8Array([
        0xff, 0xd8, 0xff, // JPEG signature
        0xe0, // APP0 marker
        // ... rest of JPEG data
      ]);

      const result = extractImagePayload(jpegData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/jpeg');
      expect(result?.bytes).toStrictEqual(jpegData);
    });
  });

  describe('GIF detection', () => {
    it('should detect GIF87a', () => {
      // GIF87a signature: 47 49 46 38 37 61
      const gifData = new Uint8Array([
        0x47, 0x49, 0x46, 0x38, 0x37, 0x61, // GIF87a signature
        // ... rest of GIF data
      ]);

      const result = extractImagePayload(gifData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/gif');
      expect(result?.bytes).toStrictEqual(gifData);
    });

    it('should detect GIF89a', () => {
      // GIF89a signature: 47 49 46 38 39 61
      const gifData = new Uint8Array([
        0x47, 0x49, 0x46, 0x38, 0x39, 0x61, // GIF89a signature
        // ... rest of GIF data
      ]);

      const result = extractImagePayload(gifData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/gif');
      expect(result?.bytes).toStrictEqual(gifData);
    });

    it('should not detect invalid GIF', () => {
      // GIF signature with invalid version byte
      const gifData = new Uint8Array([
        0x47, 0x49, 0x46, 0x38, 0x30, 0x61, // GIF version with '0' instead of '7' or '9'
        // ... rest of GIF data
      ]);

      const result = extractImagePayload(gifData);
      expect(result).toBeNull();
    });
  });

  describe('WebP detection', () => {
    it('should detect WebP image at offset 0', () => {
      // WebP signature: RIFF....WEBP
      const webpData = new Uint8Array([
        0x52, 0x49, 0x46, 0x46, // "RIFF"
        0x00, 0x00, 0x00, 0x00, // file size (placeholder)
        0x57, 0x45, 0x42, 0x50, // "WEBP"
        0x56, 0x50, 0x38, 0x20, // "VP8 "
        // ... rest of WebP data
      ]);

      const result = extractImagePayload(webpData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/webp');
      expect(result?.bytes).toStrictEqual(webpData);
    });

    it('should detect WebP image with offset', () => {
      // WebP with some padding before the signature
      const webpData = new Uint8Array([
        0x00, 0x00, 0x00, 0x00, // padding
        0x52, 0x49, 0x46, 0x46, // "RIFF"
        0x00, 0x00, 0x00, 0x00, // file size (placeholder)
        0x57, 0x45, 0x42, 0x50, // "WEBP"
        0x56, 0x50, 0x38, 0x20, // "VP8 "
      ]);

      const result = extractImagePayload(webpData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/webp');
      expect(result?.bytes).toEqual(new Uint8Array(webpData.slice(4)));
    });

    it('should not detect RIFF without WEBP', () => {
      // RIFF without WEBP signature (e.g., WAV file)
      const riffData = new Uint8Array([
        0x52, 0x49, 0x46, 0x46, // "RIFF"
        0x00, 0x00, 0x00, 0x00, // file size
        0x57, 0x41, 0x56, 0x45, // "WAVE" instead of "WEBP"
      ]);

      const result = extractImagePayload(riffData);
      expect(result).toBeNull();
    });

    it('should handle WebP file with trailing zeros', () => {
      // Actual WebP file structure (like the sample file)
      const webpData = new Uint8Array([
        0x52, 0x49, 0x46, 0x46, // "RIFF"
        0x1a, 0x0f, 0x00, 0x00, // file size: 3874 bytes
        0x57, 0x45, 0x42, 0x50, // "WEBP"
        0x56, 0x50, 0x38, 0x20, // "VP8 "
        0x0e, 0x00, 0x00, 0x00, // VP8 chunk length
        // ... rest of data
        0xde, 0x4f, 0x49, 0x64, // last bytes of actual data
        0x00, 0x00, // trailing zeros (should not be trimmed by extractImagePayload)
      ]);

      const result = extractImagePayload(webpData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/webp');
      // extractImagePayload should NOT trim trailing zeros
      expect(result?.bytes.length).toBe(webpData.length);
    });
  });

  describe('Edge cases', () => {
    it('should return null for empty data', () => {
      const result = extractImagePayload(new Uint8Array([]));
      expect(result).toBeNull();
    });

    it('should return null for data smaller than 4 bytes', () => {
      const result = extractImagePayload(new Uint8Array([0x00, 0x01, 0x02]));
      expect(result).toBeNull();
    });

    it('should return null for unknown binary data', () => {
      const unknownData = new Uint8Array([0x00, 0x01, 0x02, 0x03, 0x04, 0x05]);
      const result = extractImagePayload(unknownData);
      expect(result).toBeNull();
    });

    it('should detect GIF with offset within search range', () => {
      // GIF at offset 8 (within the 16-byte search range)
      const gifData = new Uint8Array([
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 8 bytes padding
        0x47, 0x49, 0x46, 0x38, 0x37, 0x61, // GIF87a signature
      ]);

      const result = extractImagePayload(gifData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/gif');
      expect(result?.bytes).toEqual(new Uint8Array(gifData.slice(8)));
    });
  });

  describe('AVIF detection', () => {
    it('should detect AVIF with avif brand', () => {
      // AVIF structure (ISOBMFF):
      // offset 0-3: box size (big-endian)
      // offset 4-7: "ftyp" box type
      // offset 8-11: brand ("avif" or "avis")
      const avifData = new Uint8Array([
        0x00, 0x00, 0x00, 0x20, // box size: 32 bytes
        0x66, 0x74, 0x79, 0x70, // "ftyp"
        0x61, 0x76, 0x69, 0x66, // "avif" brand
        0x00, 0x00, 0x00, 0x00, // version/flags
        // ... rest of AVIF data
      ]);

      const result = extractImagePayload(avifData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/avif');
      expect(result?.bytes).toStrictEqual(avifData);
    });

    it('should detect AVIF with avis brand', () => {
      // AVIF with "avis" brand (AVIF image sequence)
      const avifData = new Uint8Array([
        0x00, 0x00, 0x00, 0x20, // box size
        0x66, 0x74, 0x79, 0x70, // "ftyp"
        0x61, 0x76, 0x69, 0x73, // "avis" brand
        0x00, 0x00, 0x00, 0x00, // version/flags
        // ... rest of AVIF data
      ]);

      const result = extractImagePayload(avifData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/avif');
      expect(result?.bytes).toStrictEqual(avifData);
    });

    it('should not detect ftyp with non-AVIF brand', () => {
      // ISOBMFF file with different brand (e.g., MP4)
      const mp4Data = new Uint8Array([
        0x00, 0x00, 0x00, 0x20, // box size
        0x66, 0x74, 0x79, 0x70, // "ftyp"
        0x6d, 0x70, 0x34, 0x32, // "mp42" brand (not AVIF)
        0x00, 0x00, 0x00, 0x00, // version/flags
      ]);

      const result = extractImagePayload(mp4Data);
      expect(result).toBeNull();
    });

    it('should require at least 12 bytes for AVIF detection', () => {
      // Too short to be valid AVIF
      const shortData = new Uint8Array([
        0x00, 0x00, 0x00, 0x20, // box size
        0x66, 0x74, 0x79, 0x70, // "ftyp"
        0x61, 0x76, 0x69, // incomplete "avif" brand
      ]);

      const result = extractImagePayload(shortData);
      expect(result).toBeNull();
    });
  });

  describe('SVG detection', () => {
    it('should detect SVG starting with <svg', () => {
      const svgData = new TextEncoder().encode('<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40" /></svg>');
      const result = extractImagePayload(svgData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/svg+xml');
    });

    it('should detect SVG starting with XML declaration', () => {
      const svgData = new TextEncoder().encode('<?xml version="1.0" encoding="UTF-8"?><svg>...</svg>');
      const result = extractImagePayload(svgData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/svg+xml');
    });

    it('should detect SVG with leading whitespace', () => {
      const svgData = new TextEncoder().encode('   <svg>...</svg>');
      const result = extractImagePayload(svgData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/svg+xml');
    });

    it('should be case-insensitive for <svg tag', () => {
      const svgData = new TextEncoder().encode('<SVG>...</SVG>');
      const result = extractImagePayload(svgData);
      expect(result).not.toBeNull();
      expect(result?.mime).toBe('image/svg+xml');
    });

    it('should not detect general XML as SVG if <svg tag is missing in first 512 bytes', () => {
      const xmlData = new TextEncoder().encode('<?xml version="1.0"?><note><to>Tove</to></note>');
      const result = extractImagePayload(xmlData);
      expect(result).toBeNull();
    });

    it('should not detect plain text containing " <svg" late in the file', () => {
      // Create a long text that contains "<svg" after 512 bytes
      const longText = 'a'.repeat(600) + '<svg>...</svg>';
      const textData = new TextEncoder().encode(longText);
      const result = extractImagePayload(textData);
      expect(result).toBeNull();
    });
  });
});
