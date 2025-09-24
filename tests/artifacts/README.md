# Test Artifacts

This directory contains known test photos for face recognition testing.

## Photos with Faces

### Sasha Strogan
- `sasha_photo1.jpg` - Clear face photo (original: IMG_20221016_130350.JPG)
- `sasha_photo2.jpg` - Different angle/lighting (original: IMG_20221016_133143.JPG)
- `sasha_photo3.jpg` - Third photo for multi-encoding tests (original: IMG_20221017_161723.JPG)

## Photos without Faces
- `no_faces_photo1.jpg` - Landscape/object photo (original: IMG_20221018_134438.JPG)
- `no_faces_photo2.jpg` - Landscape/object photo (original: IMG_20221018_134959.JPG)
- `no_faces_photo3.jpg` - Landscape/object photo (original: IMG_20221018_163657.JPG)

## Usage

These artifacts are used by the face recognition test suite to ensure:
- Consistent test data across runs
- Reliable face detection and recognition testing
- Isolated testing without affecting production data
- Known expected outcomes for validation

## Test Scenarios

1. **Add Person**: Use sasha_photo1.jpg to add "Sasha Strogan"
2. **Recognition**: Test recognition using sasha_photo2.jpg and sasha_photo3.jpg
3. **Multiple Encodings**: Add multiple photos of same person
4. **Negative Cases**: Test no-face photos return 0 faces detected
5. **Database Operations**: Test add, list, remove operations