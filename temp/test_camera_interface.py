"""
test_camera_interface.py
========================
Test the expanded BaslerCamera interface with live preview and recording.
Mirrors MATLAB BaslerCamera_L usage.
"""

from camera import BaslerCamera
import numpy as np
import time


def test_live_preview():
    """Test: Live preview of camera stream."""
    print("\n" + "="*60)
    print(" Test: Live Preview")
    print("="*60 + "\n")

    try:
        # Create camera
        cam = BaslerCamera(exposure_us=10000)

        # Start live preview (displays video window)
        cam.startview()
        print("Live preview running. Press 'q' in the window to stop, or wait 10 seconds...\n")
        time.sleep(10)

        # Stop preview
        cam.stopview()
        print("\nPreview stopped.\n")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            cam.close()
        except:
            pass


def test_hardware_trigger_setup():
    """Test: Hardware trigger configuration."""
    print("\n" + "="*60)
    print(" Test: Hardware Trigger Setup")
    print("="*60 + "\n")

    try:
        cam = BaslerCamera(exposure_us=10000)

        # Configure for hardware-triggered acquisition
        cam.nframe(100)                  # 100 frames per trigger
        cam.framerate(100)               # 100 fps
        cam.hardwareTrigg()              # Line2 input, rising edge
        cam.exposure('Once')             # Single exposure per trigger

        print("\nCamera configured for hardware-triggered acquisition")
        print("  - Frames per trigger: 100")
        print("  - Frame rate: 100 fps")
        print("  - Trigger source: Line2 (rising edge)")
        print("  - Exposure: Once (manual)")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            cam.close()
        except:
            pass


def test_roi():
    """Test: Region of Interest configuration."""
    print("\n" + "="*60)
    print(" Test: Region of Interest (ROI)")
    print("="*60 + "\n")

    try:
        cam = BaslerCamera()

        # Set a 256x256 ROI at offset (128, 128)
        cam.roi(width=256, height=256, x_offset=128, y_offset=128)

        print("\nROI configured:")
        print("  - Size: 256x256 pixels")
        print("  - Offset: (128, 128)")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            cam.close()
        except:
            pass


def test_software_trigger():
    """Test: Software trigger mode."""
    print("\n" + "="*60)
    print(" Test: Software Trigger")
    print("="*60 + "\n")

    try:
        cam = BaslerCamera(exposure_us=10000)

        # Configure for software-triggered acquisition
        cam.softwareTrigg()
        cam.framerate(30)  # 30 fps
        cam.exposure('Continuous')  # Auto exposure

        print("\nCamera configured for software-triggered acquisition")
        print("  - Trigger source: Software")
        print("  - Frame rate: 30 fps")
        print("  - Exposure: Continuous (auto)")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            cam.close()
        except:
            pass


def test_exposure_control():
    """Test: Exposure time and mode control."""
    print("\n" + "="*60)
    print(" Test: Exposure Control")
    print("="*60 + "\n")

    try:
        cam = BaslerCamera()

        # Test different exposure modes
        cam.exposure(10000)              # Set exposure time to 10000 µs
        print("Exposure time set to 10000 µs")

        cam.exposure('Once')             # Single shot exposure
        print("Exposure mode set to 'Once'")

        cam.exposure('Continuous')       # Continuous auto exposure
        print("Exposure mode set to 'Continuous'")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            cam.close()
        except:
            pass


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" BaslerCamera Extended Interface Tests")
    print("="*60)
    print("\nNote: Make sure Basler Pylon Viewer is closed")
    print("and no other Python instances have the camera open.\n")

    # Run tests (choose which ones to run)
    try:
        # test_live_preview()
        test_hardware_trigger_setup()
        test_roi()
        test_software_trigger()
        test_exposure_control()

        print("\n" + "="*60)
        print(" All tests completed")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nFatal error: {e}")
        print("\nTroubleshooting:")
        print("1. Close Basler Pylon Viewer if it's running")
        print("2. Restart your Python kernel")
        print("3. Unplug and replug the camera USB cable")
