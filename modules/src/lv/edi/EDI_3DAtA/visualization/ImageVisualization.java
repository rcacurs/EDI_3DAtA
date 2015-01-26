package lv.edi.EDI_3DAtA.visualization;

import java.awt.Point;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.SinglePixelPackedSampleModel;
import java.awt.image.WritableRaster;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
 * 
 * @author Ričards Cacurs
 * 
 * Class providing functions for image visualization
 *
 */
public class ImageVisualization{
	
	/**
	 * Method for converting DenseMatrix64F from EJML to Java Buffered Image.
	 * The output image is normalized.
	 * 
	 * @param denseMatrix matrix containing image data
	 * @return BufferedImage java buffered image that can be drawn with swing gui
	 * framework
	 */
	public static BufferedImage convDenseMatrixToBufImage(DenseMatrix64F denseMatrix){
		int[] mask = {0xff};
		WritableRaster raster = WritableRaster.createWritableRaster(new SinglePixelPackedSampleModel(DataBuffer.TYPE_BYTE, denseMatrix.numCols, denseMatrix.numRows, mask), new Point(0, 0));
		double min_value = CommonOps.elementMin(denseMatrix);
		double max_value = CommonOps.elementMax(denseMatrix);
		for(int i=0; i<denseMatrix.numRows; i++){
			for(int j=0; j<denseMatrix.numCols; j++){
				raster.setSample(j,i, 0, 255*(denseMatrix.get(i,j)-min_value)/(max_value-min_value));
			}
		}
		BufferedImage bImage = new BufferedImage(denseMatrix.numCols, denseMatrix.numRows, BufferedImage.TYPE_BYTE_GRAY);
		bImage.setData(raster);
		return bImage;
	}
	/**
	 * Function for creating window that displays DenseMatrix64F content
	 * 
	 * @param denseMatrix matrix to be visualized
	 * @param frame object on which image is placed
	 */
	public static void imshow(DenseMatrix64F denseMatrix, JFrame frame){
		frame.getContentPane().removeAll();
		BufferedImage bImage = ImageVisualization.convDenseMatrixToBufImage(denseMatrix);
		frame.setSize(bImage.getWidth(), bImage.getHeight());
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JLabel label = new JLabel(new ImageIcon(bImage));
		JPanel jPanel = new JPanel();
		jPanel.add(label);
		frame.add(jPanel);
		frame.setVisible(true);
		frame.revalidate();
	}
}
