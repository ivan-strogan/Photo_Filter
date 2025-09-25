#!/usr/bin/env python3
"""
Focused face recognition test.

Test flow:
1. Add Elena using 1 photo (training_photo)
2. Test that identical and similar photos recognize her
3. Test that 3 no-face photos return 0 faces detected
"""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class FocusedFaceRecognitionTest(unittest.TestCase):
    """Focused face recognition test matching the specified requirements."""

    @classmethod
    def setUpClass(cls):
        """Set up test artifacts."""
        cls.test_artifacts_dir = Path(__file__).parent / "artifacts" / "photos"

        # Woman photos for graduated testing approach
        cls.training_photo = cls.test_artifacts_dir / "Woman_Photo_1.jpeg"      # Training photo
        cls.identical_photo = cls.test_artifacts_dir / "Woman_Photo_1_copy.jpeg"  # Identical copy for baseline
        cls.similar_photo2 = cls.test_artifacts_dir / "Woman_Photo_2.jpg"       # Different pose
        cls.similar_photo3 = cls.test_artifacts_dir / "Woman_Photo_3.jpg"       # Different pose

        # No face photos
        cls.no_faces_photo1 = cls.test_artifacts_dir / "no_faces_photo1.jpg"
        cls.no_faces_photo2 = cls.test_artifacts_dir / "no_faces_photo2.jpg"
        cls.no_faces_photo3 = cls.test_artifacts_dir / "no_faces_photo3.jpg"

        # Verify all test artifacts exist
        all_photos = [
            cls.training_photo, cls.identical_photo, cls.similar_photo2, cls.similar_photo3,
            cls.no_faces_photo1, cls.no_faces_photo2, cls.no_faces_photo3
        ]

        missing = [p for p in all_photos if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing test photos: {missing}")

        print(f"✅ Found all 6 test photos")

    def setUp(self):
        """Set up isolated test database for each test."""
        # Create temporary test database
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="face_test_"))
        self.test_people_db_file = self.test_data_dir / "people_database.json"

        # Initialize face recognition components
        from src.face_recognizer import FaceRecognizer
        from src.people_database import PeopleDatabase

        self.test_face_cache_file = self.test_data_dir / "face_cache.json"
        self.people_db = PeopleDatabase(database_file=self.test_people_db_file)
        self.face_recognizer = FaceRecognizer(
            detection_model="hog",
            recognition_tolerance=0.6,
            cache_file=self.test_face_cache_file,
            people_database=self.people_db
        )

    def tearDown(self):
        """Clean up test database."""
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def test_complete_face_recognition_workflow(self):
        """Complete test: train with 1 photo, recognize in 2 others, verify no-face photos."""
        print("\n🎯 Complete Face Recognition Test")
        print("=" * 40)

        # Step 1: Train with training photo and add Elena
        print("📸 Step 1: Training with training photo...")

        # First verify the training photo has a face
        training_result = self.face_recognizer.detect_faces(self.training_photo)
        self.assertIsNone(training_result.error, "Training photo should not error")
        self.assertGreater(training_result.faces_detected, 0, "Training photo should have faces")
        print(f"   Training photo: {training_result.faces_detected} faces detected")

        # Add Elena using the training photo
        success = self.face_recognizer.add_person("Elena Rodriguez", [self.training_photo])
        self.assertTrue(success, "Should successfully add Elena")

        # Verify Elena was added
        known_people = self.face_recognizer.list_known_people()
        self.assertEqual(len(known_people), 1, "Should have 1 person")
        self.assertIn("Elena Rodriguez", known_people, "Should contain Elena Rodriguez")
        print(f"   ✅ Added Elena to database")

        # Step 2: Test graduated recognition approach
        print("\n🔍 Step 2: Testing graduated recognition approach...")

        test_photos = [
            (self.identical_photo, "identical_photo"),
            (self.similar_photo2, "similar_photo2")
        ]

        for photo, photo_name in test_photos:
            result = self.face_recognizer.detect_faces(photo)
            people_detected = result.get_people_detected()

            print(f"   {photo_name}: {result.faces_detected} faces, recognized: {people_detected}")

            self.assertIsNone(result.error, f"Should not error on {photo_name}")
            self.assertGreater(result.faces_detected, 0, f"Should detect faces in {photo_name}")
            self.assertIn("Elena Rodriguez", people_detected, f"Should recognize Elena in {photo_name}")

        print("   ✅ Elena recognized in both test photos")

        # Step 3: Test no-face photos return 0 faces
        print("\n🚫 Step 3: Testing no-face photos...")

        no_face_photos = [
            (self.no_faces_photo1, "no_faces_photo1"),
            (self.no_faces_photo2, "no_faces_photo2"),
            (self.no_faces_photo3, "no_faces_photo3")
        ]

        for photo, photo_name in no_face_photos:
            result = self.face_recognizer.detect_faces(photo)
            people_detected = result.get_people_detected()

            print(f"   {photo_name}: {result.faces_detected} faces, people: {people_detected}")

            self.assertIsNone(result.error, f"Should not error on {photo_name}")
            self.assertEqual(result.faces_detected, 0, f"Should detect 0 faces in {photo_name}")
            self.assertEqual(len(people_detected), 0, f"Should detect 0 people in {photo_name}")

        print("   ✅ All no-face photos correctly returned 0 faces")

        # Step 4: Test person removal
        print("\n🗑️ Step 4: Testing person removal...")

        # Verify Elena is currently recognized
        pre_removal_result = self.face_recognizer.detect_faces(self.identical_photo)
        pre_removal_people = pre_removal_result.get_people_detected()
        self.assertIn("Elena Rodriguez", pre_removal_people, "Elena should be recognized before removal")
        print(f"   Before removal: {pre_removal_people}")

        # Remove Elena from database
        removed = self.face_recognizer.remove_person("Elena Rodriguez")
        self.assertTrue(removed, "Should successfully remove Elena")

        # Clear face cache to ensure fresh detection
        self.face_recognizer.face_cache.clear()

        # Verify database is now empty
        known_people = self.face_recognizer.list_known_people()
        self.assertEqual(len(known_people), 0, "Database should be empty after removal")
        print(f"   ✅ Elena removed from database (cache cleared)")

        # Test that Elena is no longer recognized in the same photos
        test_photos_after_removal = [
            (self.identical_photo, "identical_photo"),
            (self.similar_photo2, "similar_photo2")
        ]

        for photo, photo_name in test_photos_after_removal:
            result = self.face_recognizer.detect_faces(photo)
            people_detected = result.get_people_detected()

            print(f"   After removal - {photo_name}: {result.faces_detected} faces, recognized: {people_detected}")

            self.assertGreater(result.faces_detected, 0, f"Should still detect faces in {photo_name}")
            self.assertNotIn("Elena Rodriguez", people_detected, f"Should NOT recognize Elena in {photo_name} after removal")
            self.assertEqual(len(people_detected), 0, f"Should recognize no people in {photo_name} after removal")

        print("   ✅ Elena no longer recognized after removal")

        # Step 5: Summary
        print("\n📊 Test Summary:")
        stats = self.face_recognizer.get_statistics()
        print(f"   People in database: {stats['known_people']}")
        print(f"   Total encodings: {stats['total_known_encodings']}")
        print("   ✅ Training: 1 photo → person added")
        print("   ✅ Recognition: 2 photos → Elena identified")
        print("   ✅ No faces: 3 photos → 0 faces detected")
        print("   ✅ Removal: Person removed → no longer recognized")


def run_test():
    """Run the focused face recognition test."""
    print("🧪 Focused Face Recognition Test")
    print("📋 Test Plan:")
    print("   1. Train with training_photo")
    print("   2. Recognize Elena in identical and similar photos")
    print("   3. Verify no faces in 3 no-face photos")
    print("   4. Remove Elena and verify she's no longer recognized")
    print()

    # Check face recognition availability
    try:
        import face_recognition
        print("✅ face_recognition library available")
    except ImportError:
        print("❌ face_recognition library not available")
        return False

    # Run the test
    suite = unittest.TestSuite()
    suite.addTest(FocusedFaceRecognitionTest('test_complete_face_recognition_workflow'))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 Face recognition test PASSED!")
        print("✅ System working correctly:")
        print("   • Can train with 1 photo")
        print("   • Can recognize person in other photos")
        print("   • Correctly identifies photos with no faces")
        print("   • Can remove person and stop recognizing them")
    else:
        print("❌ Face recognition test FAILED")
        if result.failures:
            print("Failures:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback}")
        if result.errors:
            print("Errors:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)