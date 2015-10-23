package lv.edi.EDI_3DAtA.opencvcudainterface;

public class Compute {
	
	//public native double[] gaussianBlur(double[] input, int rows, int cols);
	//public native double[] segmentBloodVessels(double[] input, int rows, int cols, double[] codes, double[] means, int patchSize, int numberOfFilters, double[] model, double[] scaleparamsMeans, double[] scaleparamsSd, double[] imageMask);
	public native void test();
	public native double[] segmentBloodVessels(double[] input, int r1, int c1,
			                                   double[] codes, int r2, int c2,
			                                   double[] means, int r3, int c3,
			                                   double[] scalesMean, int r4, int c4,
			                                   double[] model, int r5, int c5,
			                                   double[] scalesSd, int r6, int c6);
	public Compute() throws UnsatisfiedLinkError{
		//String libPath = System.getProperty("java.library.path");
		//System.out.println(libPath);
		System.loadLibrary("computeCudaInterface");

	}
}
