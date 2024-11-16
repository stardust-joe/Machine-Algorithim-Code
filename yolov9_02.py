from ultralytics import YOLO
model=YOLO("700i_100e.pt")
image_list=["2024-10-24_125332_Frame_23.png","2024-11-07_132105_Frame_2192.png",
            "2024-10-01_110808_Frame_528.png",
            "2024-11-07_133538_Frame_753.png","2024-11-07_133538_Frame_601.png",
            "2024-11-01_162646_Frame_653.png","2024-10-01_110808_Frame_233.png",
            "2024-11-01_161514_Frame_11.png","2024-10-01_112116_Frame_592.png",
            "2024-10-01_113725_Frame_9.png","Cc15.jpg","cc10.jpg","c43.jpg","sq1.jpg","sq19.jpg","sq8.jpg",
            "sq3.jpg","sq4.jpg","C9.jpg","2024-11-01_161000_Frame_295.jpg","cc10.jpg","C4.jpg","C12.jpg","2024-11-07_132500_Frame_117.jpg","2024-11-14_161734_Frame_448.png","2024-11-14_162608_Frame_382.png"]
results=model.predict(image_list, imgsz=640, conf=0.5, save=True, show=True)
#"2024-11-07_132105_Frame_2192.png"
