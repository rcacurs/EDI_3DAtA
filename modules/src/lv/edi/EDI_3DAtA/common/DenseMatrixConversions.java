package lv.edi.EDI_3DAtA.common;


import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class DenseMatrixConversions 
{
	
	/**
     * Convert the input 2D array of integers to the DenseMatrix64F format
     * @param inputArray - input array
     * @return - array in the DenseMatrix64F format
     */
	public static DenseMatrix64F convertArrayToDense ( int[][] inputArray )
	{
		// Type cast the Array of integers to the array of doubles:		
		// double[][] doubleArray = Arrays.copyOf( inputArray, inputArray.length * inputArray[0].length, double[][].class );
		
		double[][] doubleArray = new double[inputArray.length][inputArray[0].length];
		
		for ( int row = 0; row < inputArray.length; row++ )
		{
			for ( int col = 0; col < inputArray[0].length; col++ )
			{
				doubleArray[row][col] = ( double ) inputArray[row][col];
			}
		}
		
		return new DenseMatrix64F( doubleArray );
		
	}
	
	/**
     * Convert DenseMatrix64F to CSV string/array
     * @param matrix - input matrix of DenseMatrix64F type
     * @return - the string containing the elements of the input matrix
     */
	public static String DenseMatrixToCSVString ( DenseMatrix64F matrix ) 
	{
		
		StringBuilder csvStringBuilder = new StringBuilder();
		
		for(int i=0; i<matrix.numRows; i++)
		{
			for(int j=0; j<matrix.numCols - 1; j++)
			{
				csvStringBuilder.append( matrix.get(i,j) );
				csvStringBuilder.append( "," );
				//csvStringBuilder.append(matrix.get(i,j)+",");
			}
			csvStringBuilder.append( matrix.get(i, matrix.numCols-1) );
			csvStringBuilder.append( "\n" );
			
			// csvStringBuilder.append(matrix.get(i, matrix.numCols-1)+"\n");
		}
		return csvStringBuilder.toString();
		
	}
	
	/**
     * save dense matrix to CSV file
     * @param matrix - matrix to save in DenseMatrix64F format
     * @param fileName - file name
     */
	public static void saveDenseMatrixToCSV( DenseMatrix64F matrix, String fileName )
	{
		int maxNumEl = 1000000; // if number of elements in the array is larger than this number save it in parts
		int partsToSplit = 10; // split the matrix into "partsToSplit" parts
		if ( matrix.getNumElements() > maxNumEl ) // if matrix is big split into parts and save
		{
			
			if ( matrix.numRows > matrix.numCols ) // split "horizontally" if true
			{
				try 
				{
					PrintWriter outFile = new PrintWriter(fileName + ".csv"); // new PrintWriter
					
					int numRowsPerBlock = matrix.numRows / partsToSplit; // number of rows in the sub-matrix
					int startY = 0; // Y coordinate of the sub-matrix start
					int endY = 0; // Y coordinate of the sub-matrix end
					
					for ( int i = 0; i < partsToSplit; i++ )
					{
						startY = i * numRowsPerBlock;
						endY = (i+1) * numRowsPerBlock;
						// save sub-matrix:
						outFile.write(    DenseMatrixConversions.DenseMatrixToCSVString(  CommonOps.extract(matrix, startY, endY, 0, matrix.numCols)  )    );
					}
					
					if ( matrix.numRows > endY ) // save remaining part of the array
					{
						startY = endY;
						endY = matrix.numRows;
						outFile.write(    DenseMatrixConversions.DenseMatrixToCSVString(  CommonOps.extract(matrix, startY, endY, 0, matrix.numCols)  )    );
					}
					
					outFile.close(); // close the PrintWriter
				} 
				catch (Exception ex) 
				{
					System.out.println("File not saved");
				}
				
			}
			else // add vertical splitting later if needed (for now - save without splitting)
			{
				try 
				{
					PrintWriter outFile = new PrintWriter(fileName + ".csv");
					outFile.write( DenseMatrixConversions.DenseMatrixToCSVString(matrix) ); // write(char[] buf) - writes an array of characters.
					outFile.close();
				} 
				catch (Exception ex) 
				{
					System.out.println("File not saved");
				}
			}
			
		}
		else // if matrix is small save without splitting
		{
			
			try 
			{
				PrintWriter outFile = new PrintWriter(fileName + ".csv");
				outFile.write( DenseMatrixConversions.DenseMatrixToCSVString(matrix) ); // write(char[] buf) - writes an array of characters.
				outFile.close();
			} 
			catch (IOException ex) 
			{
				System.out.println("File not saved");
			}
			
		}
		
	}
	
	/**
     * load the data from CSV file into the output matrix of the DenseMatrix64F type
     * @param fileName - file name
     * @return loaded data in the DenseMatrix64F format
     */
	public static DenseMatrix64F loadCSVtoDenseMatrix( String fileName )
	{
		try 
		{
			FileReader reader = new FileReader( fileName + ".csv" ); // define FileReader
			BufferedReader buffer = new BufferedReader(reader); // define BufferedReader
			
			String tempString = ""; // the string to store the line from the CSV file 
			ArrayList <double[]> storeArrayList = new ArrayList<double[]>(); // define the ArrayList to store the arrays of doubles
			int countMax = 0; // variable to store the number of Tokens in the current string
			
			while ( ( tempString = buffer.readLine() ) != null ) // while the file has more lines to read
			{
				StringTokenizer tokenizer = new StringTokenizer(tempString, ","); // define the StringTokenizer and the delimiter
				
				countMax = tokenizer.countTokens(); // the number of tokens in the string
				double[] rowFromCSV = new double[countMax]; // vector to store the numbers from the string
				
				for ( int i = 0; i < countMax; i++ )
				{
					rowFromCSV[i] = Double.parseDouble( tokenizer.nextToken() ); // read the tokens and convert them to double
				}
				storeArrayList.add(rowFromCSV); // save the array to the ArrayList
			}
			
			double[][] outputArray = new double[storeArrayList.size()][countMax]; // double 2D array to store the data from the ArrayList
			double[] rowFromCSV = new double[countMax]; // vector to store the current row from the ArrayList
			for ( int row = 0; row < outputArray.length; row++ )
			{
				rowFromCSV = storeArrayList.get(row); // get current row
				
				for ( int col = 0; col < outputArray[0].length; col++ )
				{
					outputArray[row][col] = rowFromCSV[col]; // fill the output array
				}
			}
			
			try {
				buffer.close(); // close the BufferedReader. We only need to close the outer wrapper
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			return new DenseMatrix64F(outputArray); // return the matrix
			
		} 
		catch (IOException ex) 
		{
			System.out.println("File not loaded");
			return null; // return the null pointer
		}
	}
	
	/**
     * load the data from CSV file into the output matrix of the DenseMatrix64F type
     * @param is - input stream file representing .csv file
     * @return loaded data in the DenseMatrix64F format
     */
	public static DenseMatrix64F loadCSVtoDenseMatrixFromInputStream( InputStream is )
	{
		try 
		{
			InputStreamReader reader = new InputStreamReader( is ); // define FileReader
			BufferedReader buffer = new BufferedReader(reader); // define BufferedReader
			
			String tempString = ""; // the string to store the line from the CSV file 
			ArrayList <double[]> storeArrayList = new ArrayList<double[]>(); // define the ArrayList to store the arrays of doubles
			int countMax = 0; // variable to store the number of Tokens in the current string
			
			while ( ( tempString = buffer.readLine() ) != null ) // while the file has more lines to read
			{
				StringTokenizer tokenizer = new StringTokenizer(tempString, ","); // define the StringTokenizer and the delimiter
				
				countMax = tokenizer.countTokens(); // the number of tokens in the string
				double[] rowFromCSV = new double[countMax]; // vector to store the numbers from the string
				
				for ( int i = 0; i < countMax; i++ )
				{
					rowFromCSV[i] = Double.parseDouble( tokenizer.nextToken() ); // read the tokens and convert them to double
				}
				storeArrayList.add(rowFromCSV); // save the array to the ArrayList
			}
			
			double[][] outputArray = new double[storeArrayList.size()][countMax]; // double 2D array to store the data from the ArrayList
			double[] rowFromCSV = new double[countMax]; // vector to store the current row from the ArrayList
			for ( int row = 0; row < outputArray.length; row++ )
			{
				rowFromCSV = storeArrayList.get(row); // get current row
				
				for ( int col = 0; col < outputArray[0].length; col++ )
				{
					outputArray[row][col] = rowFromCSV[col]; // fill the output array
				}
			}
			
			try {
				buffer.close(); // close the BufferedReader. We only need to close the outer wrapper
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			return new DenseMatrix64F(outputArray); // return the matrix
			
		} 
		catch (IOException ex) 
		{
			System.out.println("File not loaded");
			return null; // return the null pointer
		}
	}
	
}
