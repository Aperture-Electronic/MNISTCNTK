using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CNTK;

namespace MNISTCNTKClient
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void CanDraw_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                System.Windows.Point point = e.GetPosition(canDraw);
                RectangleGeometry pt = new RectangleGeometry(new Rect(point, new System.Windows.Size(5, 5)));
                System.Windows.Shapes.Path path = new System.Windows.Shapes.Path
                {
                    Stroke = System.Windows.Media.Brushes.Black,
                    StrokeThickness = 5,
                    Data = pt
                };
                canDraw.Children.Add(path);
            }
        }

        private void BtnClear_Click(object sender, RoutedEventArgs e)
        {
            canDraw.Children.Clear();
        }

        private void BtnRegress_Click(object sender, RoutedEventArgs e)
        {
            // Get and convert the bitmap drawed
            // Render the bitmap
            RenderTargetBitmap renderBitmap = new RenderTargetBitmap((int)canDraw.ActualWidth, (int)canDraw.ActualHeight, 96, 96, PixelFormats.Default);
            renderBitmap.Render(canDraw);

            // Convert it to bitmap object
            Bitmap bitmap = null;
            using (MemoryStream stream = new MemoryStream())
            {
                BitmapEncoder encoder = new BmpBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(renderBitmap));
                encoder.Save(stream);
                bitmap = new Bitmap(stream);
            }

            // Resize the bitmap to MNIST image size (28 x 28 = 784 pixels)
            Bitmap MNIST_bitmap = new Bitmap(28, 28);
            using (Graphics graphics = Graphics.FromImage(MNIST_bitmap))
            {
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.DrawImage(bitmap, new System.Drawing.Rectangle(0, 0, 28, 28), new System.Drawing.Rectangle(0, 0, bitmap.Width - 1, bitmap.Height - 1), GraphicsUnit.Pixel);
            }

            // Load trained model for CNTK function
            Function mnist_model = Function.Load("model/trained-model.mld", DeviceDescriptor.CPUDevice);
            Variable imageInput = mnist_model.Arguments[0];
            Variable labelOutput = mnist_model.Outputs.Single(o => o.Name == "classifierOutput");
            Dictionary<Variable, Value> outputDataMap = new Dictionary<Variable, Value>();
            Variable outputVar = mnist_model.Output;
            outputDataMap.Add(outputVar, null);

            // Read and convert the pixels to model input
            NDShape input_shape = imageInput.Shape;
            List<float> input_pixels = new List<float>();
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    input_pixels.Add((MNIST_bitmap.GetPixel(x, y).ToArgb() == -1) ? 0 : 255);
                    if ((x == 0) || (y == 0)) input_pixels[input_pixels.Count - 1] = 0;
                    Console.Write(input_pixels[input_pixels.Count - 1] == 255 ? 1 : 0);
                }
                Console.WriteLine();
            }

            Dictionary<Variable, Value> inputDataMap = new Dictionary<Variable, Value>();
            Value mnist_input = Value.CreateBatch(input_shape, input_pixels, DeviceDescriptor.CPUDevice);
            inputDataMap.Add(imageInput, mnist_input);

            // Evaluate the result
            mnist_model.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);

            // Get evaluate result as dense output
            Value outputVal = outputDataMap[outputVar];
            IList<IList<float>> outputData = outputVal.GetDenseData<float>(outputVar);

            IList<float> result = outputData.First();
            int r = result.IndexOf(result.Max());

            // Output the result
            lblResult.Content = $"{r}";
        }
    }
}
