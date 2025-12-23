
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-311"))
sys.path.append(os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-39"))

try:
    import pyopenlpt
    print("pyopenlpt imported")
    if hasattr(pyopenlpt, 'Camera'):
        print("Camera class found")
        # print(dir(pyopenlpt.Camera))
        
        # Try loading
        try:
            cam = pyopenlpt.Camera("test/inputs/test_STB/camFile/cam1.txt")
            print("Camera loaded successfully")
            print(f"Dir(cam): {dir(cam)}")
            
            p = None
            if hasattr(cam, '_type'):
                print(f"Type: {cam._type}") # Enum CameraType
                if cam._type == 0: # PINHOLE
                    p = cam.pinhole_param
            elif hasattr(cam, 'pinhole_param'):
                p = cam.pinhole_param
            
            if p:
                print("Param found")
                print(f"Attributes of param: {dir(p)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to load camera: {e}")
            
    else:
        print("Camera class NOT found")
except ImportError as e:
    print(f"Import failed: {e}")
