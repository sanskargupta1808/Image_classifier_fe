from PIL import Image
from PIL.ExifTags import TAGS
import piexif

def extract_metadata_for_display(path):
    """Extract metadata for user display (not used in classification)"""
    metadata = {'camera': {}, 'total_tags': 0}
    
    try:
        # Try PIL first
        img = Image.open(path)
        exif = img.getexif()
        
        if exif:
            camera_info = {}
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ['Make', 'Model', 'Software', 'DateTime', 'DateTimeOriginal',
                          'ExposureTime', 'FNumber', 'ISOSpeedRatings', 'FocalLength', 
                          'Flash', 'WhiteBalance', 'ExposureMode', 'SceneCaptureType']:
                    camera_info[tag] = str(value)
            
            metadata['camera'] = camera_info
            metadata['total_tags'] = len(exif)
        
        # If no metadata found with PIL, try piexif for more comprehensive extraction
        if not metadata['camera']:
            try:
                exif_dict = piexif.load(path)
                camera_info = {}
                
                # Check 0th IFD (main image)
                if "0th" in exif_dict:
                    ifd = exif_dict["0th"]
                    if piexif.ImageIFD.Make in ifd:
                        camera_info['Make'] = ifd[piexif.ImageIFD.Make].decode('utf-8')
                    if piexif.ImageIFD.Model in ifd:
                        camera_info['Model'] = ifd[piexif.ImageIFD.Model].decode('utf-8')
                    if piexif.ImageIFD.Software in ifd:
                        camera_info['Software'] = ifd[piexif.ImageIFD.Software].decode('utf-8')
                
                # Check Exif IFD
                if "Exif" in exif_dict:
                    ifd = exif_dict["Exif"]
                    if piexif.ExifIFD.DateTimeOriginal in ifd:
                        camera_info['DateTimeOriginal'] = ifd[piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
                    if piexif.ExifIFD.ExposureTime in ifd:
                        camera_info['ExposureTime'] = str(ifd[piexif.ExifIFD.ExposureTime])
                    if piexif.ExifIFD.FNumber in ifd:
                        camera_info['FNumber'] = str(ifd[piexif.ExifIFD.FNumber])
                    if piexif.ExifIFD.ISOSpeedRatings in ifd:
                        camera_info['ISO'] = str(ifd[piexif.ExifIFD.ISOSpeedRatings])
                
                metadata['camera'] = camera_info
                metadata['total_tags'] = len(camera_info)
                
            except:
                pass
                
    except Exception:
        pass
    
    return metadata
