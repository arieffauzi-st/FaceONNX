using System;
using System.Drawing;
using System.IO;
using UMapx.Visualization;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace FaceONNX
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("FaceONNX: Face detection");
           
            using var faceDetector = new FaceDetector(0.95f, 0.5f);
            using var painter = new Painter()
            {
                BoxPen = new Pen(Color.Yellow, 4),
                Transparency = 0,
            };

            using var capture = new VideoCapture();

            string windowName = "Face Detection";
            CvInvoke.NamedWindow(windowName, WindowFlags.AutoSize);

            while (true) 
            {
                using var frame = capture.QueryFrame();
                if (frame != null)
                {
                    
                    Image<Bgr, byte> img = frame.ToImage<Bgr, byte>();
                    Bitmap bitmap = img.ToBitmap();

                    var output = faceDetector.Forward(bitmap);

                    foreach (var rectangle in output)
                    {
                        var paintData = new PaintData()
                        {
                            Rectangle = rectangle,
                            Title = string.Empty
                        };
                        using var graphics = Graphics.FromImage(bitmap);
                        painter.Draw(graphics, paintData);
                    }

                    // Convert the modified Bitmap back to Image<Bgr, byte>
                    Image<Bgr, byte> processedImg = bitmap.ToImage<Bgr, byte>();

                    // Display the image in the window.
                    CvInvoke.Imshow(windowName, processedImg);
                    if (CvInvoke.WaitKey(1) == 27) // Break loop on 'ESC' key press
                        break;

                    //bitmap.Save("WebcamFrame.png"); // Simpan frame sebagai gambar. Anda mungkin ingin mengubah ini.
                    Console.WriteLine($"Image: [Webcam frame] --> detected [{output.Length}] faces ");
                    //Console.WriteLine($"Image: [Webcam frame] --> detected [{output.Length}] faces with last confidence {faceDetector.LastConfidence}");
                }
            }
            CvInvoke.DestroyWindow(windowName);
        }
    }
}
    
