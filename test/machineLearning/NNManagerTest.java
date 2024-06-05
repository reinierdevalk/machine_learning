package machinelearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import junit.framework.TestCase;
import machinelearning.NNManager;
import machinelearning.NNManager.ActivationFunction;

import org.encog.ml.data.basic.BasicMLDataSet;

public class NNManagerTest extends TestCase {
  
	private List <Double> summedOutputs;
	
	protected void setUp() throws Exception {
		super.setUp();
	}

	protected void tearDown() throws Exception {
		super.tearDown();
	}
	
//	public void testCreateTrainingExamples() {
//		fail("Not yet implemented");
//	}
//
//	public void testTrainNetwork() {
//		fail("Not yet implemented");
//	}
//
//	public void testLoadWeightsXML() {
//		fail("Not yet implemented");
//	}
//
//	public void testStoreWeightsXML() {
//		fail("Not yet implemented");
//	}


	public void testNetwork() {	
		List<Double> features = Arrays.asList(new Double[]{0.4, -0.1, 0.3, 0.2, 0.5});
		int in = features.size();
		int hid = 3;
		int out = 3;
		// Create a network with randomly initialised weights and evaluate it
		NNManager netMan = new NNManager();
		netMan.initialiseNetwork(ActivationFunction.SIGMOID, new Integer[]{in, hid, out});
		netMan.initWeights(null);
		double[] outp1 = netMan.evalNetwork(features);

		// Create a second network, initialise it with the same weights, and evaluate it 
		NNManager netMan2 = new NNManager();
		netMan2.initialiseNetwork(ActivationFunction.SIGMOID, new Integer[]{in, hid, out});
		netMan2.initWeights(netMan.getNetworkWeights());
		double[] outp2 = netMan2.evalNetwork(features);

		assertEquals(outp1.length, outp2.length);
		for (int i = 0; i < outp1.length; i++) {
			assertEquals(outp1[i], outp2[i]);
		}
//		assert(outp1 == outp2);
//		fail("no way");
	}


	// OLD STUFF:
	// TODO
	private void testStability() {
//		// 1. Create features, labels, and BasicMLDataSet
//		// a. The large set
//		List<List<Double>> features = getLargeDataSet().get(0);
//		List<List<Double>> labels = getLargeDataSet().get(1);
//		List<List<List<Double>>> chordLabels = null;
//		
//		// b. The small set (sizeOfDataSet must be a multiple of 4)
//		int sizeOfDataSet = 500;
////		List<List<Double>> features = getSmallDataSet(sizeOfDataSet/4).get(0);
////		List<List<Double>> labels = getSmallDataSet(sizeOfDataSet/4).get(1);
//		
//		System.out.println(features.size());
//		System.out.println(labels.size());
//		
//		EncogNNManager networkManager = new EncogNNManager(-1, -1, features.get(0).size(), labels.get(0).size());
////		networkManager.createTrainingExamplesNoteToNote(features, labels, false, false);
//		  	 
//	  // 2. Initialise the weights
////    networkManager.initWeightsRandomly();
////	  StableFilesLoader stableFilesLoader = new StableFilesLoader();
//	  List<Double> storedStableWeights = null; //stableFilesLoader.loadStableWeightsXML();
////	    networkManager.initWeightsFromList(storedStableWeights);
////	    List<Double> bestWeights = networkManager.loadWeightsXML();
//	  
//	  
//	  int runs = 10;
//	  int cycles = 50; 
//	  
//	  // TEST 1. Train the NN with lambda = 0.003 and regularisation
//	  double lambda = 0.003;
//	  boolean regularise = true;
//	  System.out.println("\n... performing Test 1 (lambda == 0.003; regularisation) ...");
//	  List<List<List<Double>>> outputsWithLambda = testNN(networkManager, storedStableWeights, runs, cycles,
//	  	networkManager.dataset, features, labels, lambda, regularise);
//	  List<Double> sumsTest1 = summedOutputs;
//	  
//	  // TEST 2. Train the NN with lambda = 0.000 and regularisation
//		lambda = 0.000;
//		regularise = true;
//		System.out.println("\n... performing Test 2 (lambda == 0; regularisation) ...");
//		List<List<List<Double>>> outputsWithLambdaZero = testNN(networkManager, storedStableWeights, runs, cycles,
//				networkManager.dataset, features, labels, lambda, regularise);
//		List<Double> sumsTest2 = summedOutputs;
//		
//	  // TEST 3. Train the NN with lambda = 0.003 and no regularisation 
//		lambda = 0.003;
//		regularise = false;
//		System.out.println("\n... performing Test 3 (lambda == 0.003; no regularisation) ...");
//		List<List<List<Double>>> outputsWithLambdaNoReg = testNN(networkManager, storedStableWeights, runs, cycles,
//				networkManager.dataset, features, labels, lambda, regularise);
//		List<Double> sumsTest3 = summedOutputs;
//		
//	  // TEST 4. Train the NN with lambda = 0.000 and no regularisation 
//		lambda = 0;
//		regularise = false;
//		System.out.println("\n... performing Test 4 (lambda == 0; no regularisation) ...");
//		List<List<List<Double>>> outputsWithLambdaZeroNoReg = testNN(networkManager, storedStableWeights, runs, cycles,
//				networkManager.dataset, features, labels, lambda, regularise);
//		List<Double> sumsTest4 = summedOutputs;
//		
//	  // TEST 5. Now that it is asserted that the outputs of Tests 1, 2, 3, and 4 are the same for each individual run,
//    // they must be compared mutually. 
//	  // a. Quick visual test: compare the summed sums
//    double summedSummedOutputsTest1 = 0;
//    double summedSummedOutputsTest2 = 0;
//    double summedSummedOutputsTest3 = 0;
//    double summedSummedOutputsTest4 = 0;
//    for (int i = 0; i < summedOutputs.size(); i++) {
//    	summedSummedOutputsTest1 += sumsTest1.get(i);
//    	summedSummedOutputsTest2 += sumsTest2.get(i);
//    	summedSummedOutputsTest3 += sumsTest3.get(i);
//    	summedSummedOutputsTest4 += sumsTest4.get(i);
//    }
//    System.out.println("\nsummedSummedOutputsTest1:");
//    System.out.println(summedSummedOutputsTest1);
//    System.out.println("\nsummedSummedOutputsTest2:");
//    System.out.println(summedSummedOutputsTest2);
//    System.out.println("\nsummedSummedOutputsTest3:");
//    System.out.println(summedSummedOutputsTest3);
//    System.out.println("\nsummedSummedOutputsTest4:");
//    System.out.println(summedSummedOutputsTest4);
//      
//	  // Assert equality
////    assertEquals((summedSummedOutputsTest1 < summedSummedOutputsTest2), true);
////    assertEquals((summedSummedOutputsTest1 < summedSummedOutputsTest3), true);
////    assertEquals((summedSummedOutputsTest1 < summedSummedOutputsTest4), true);
//    assertFalse(summedSummedOutputsTest1 == summedSummedOutputsTest2);
//    assertFalse(summedSummedOutputsTest1 == summedSummedOutputsTest3);
//    assertFalse(summedSummedOutputsTest1 == summedSummedOutputsTest4);
//    assertEquals(summedSummedOutputsTest2, summedSummedOutputsTest3);
//    assertEquals(summedSummedOutputsTest2, summedSummedOutputsTest4);
//    
//    
//    // b. Compare the outputs 
//    for (int k = 0; k < runs; k++) {
//      for (int i = 0; i < features.size(); i++) {
//      	for (int j = 0; j < labels.get(0).size(); j++) {
//	      	double outputOfRun1 = outputsWithLambda.get(0).get(i).get(j);
//	      	double outputOfRun2 = outputsWithLambdaZero.get(0).get(i).get(j);
//	      	double outputOfRun3 = outputsWithLambdaNoReg.get(0).get(i).get(j);
//	      	double outputOfRun4 = outputsWithLambdaZeroNoReg.get(0).get(i).get(j);
//	      	// The output of (any of the runs of) Test 1 must be different from the output of (any of the runs of)
//	      	// Test 2, 3, and 4    
//	      	assertEquals(outputOfRun1 == outputOfRun2, false);
//  	    	assertEquals(outputOfRun1 == outputOfRun3, false);
//  	    	assertEquals(outputOfRun1 == outputOfRun4, false);
//	      	// The output of (any of the runs of) Test 2 must be equal to the output of (any of the runs of) Test 3
//  	    	// and 4 
//  	    	assertEquals(outputOfRun2, outputOfRun3);
//  	    	assertEquals(outputOfRun2, outputOfRun4);
//	      }
//	    }
//    }
	}
	
	// Returns the original large dataSet
	// TODO
	private List<List<List<Double>>> getLargeDataSet() {
		List<List<List<Double>>> largeDataSet = new ArrayList<List<List<Double>>>(); 
//		StableFilesLoader sfl = new StableFilesLoader(); 
//		List<List<Double>> trainingFeatures = sfl.getStableFeatures(DataPreparer.setA).get(0);
		List<List<Double>> trainingFeatures = null; //sfl.getStableFeatures(1).get(0); // TODO was sfl.getStableFeatures(DataPreparer.setA).get(0);
		List<List<Double>> trainingLabels = null; // sfl.getStableLabels().get(0);
		
//		GeneralManagerGraz.loadSetAFeatures();
//		GeneralManagerGraz.loadLabels();
//		List<List<Double>> trainingFeatures = GeneralManagerGraz.trainingSetAFeatures;
//		List<List<Double>> trainingLabels = GeneralManagerGraz.trainingSetLabels;
		
		largeDataSet.add(trainingFeatures);
		largeDataSet.add(trainingLabels);
		
		return largeDataSet;
	}
	
	
	// Returns a small dataSet
	// TODO
	private List<List<List<Double>>> getSmallDataSet(int sizeOfDataSet) {
		List<List<List<Double>>> smallDataSet = new ArrayList<List<List<Double>>>(); 
		
		// Make some features and some labels
		List<Double> features1 = new ArrayList<Double>();
		features1.add(0.1); features1.add(-0.1); features1.add(0.33); features1.add(-0.7); features1.add(0.2); 
		List<Double> features2 = new ArrayList<Double>();
		features2.add(-0.1); features2.add(0.76); features2.add(-0.12); features2.add(0.5); features2.add(-0.43);
		List<Double> features3 = new ArrayList<Double>();
		features3.add(0.91); features3.add(-0.33); features3.add(-0.42); features3.add(0.76); features3.add(-0.1);
		List<Double> features4 = new ArrayList<Double>();
		features4.add(-0.71); features4.add(-0.46); features4.add(0.82); features4.add(-0.25); features4.add(0.53);
		
		List<Double> label1 = new ArrayList<Double>();
		label1.add(0.0); label1.add(0.0); label1.add(1.0); label1.add(0.0); label1.add(0.0);
		List<Double> label2 = new ArrayList<Double>();
		label2.add(1.0); label2.add(0.0);	label2.add(0.0); label2.add(0.0);	label2.add(0.0);
		List<Double> label3 = new ArrayList<Double>();
		label3.add(0.0); label3.add(1.0); label3.add(0.0); label3.add(0.0); label3.add(0.0);
		List<Double> label4 = new ArrayList<Double>();
		label4.add(0.0); label4.add(0.0);	label4.add(0.0); label4.add(1.0);	label4.add(0.0);
		
		// Make the list of features and the list of labels
		List<List<Double>> trainingFeatures = new ArrayList<List<Double>>();
		List<List<Double>> trainingLabels = new ArrayList<List<Double>>();
		for (int i = 0; i < sizeOfDataSet; i++) {
	    trainingFeatures.add(features1);
	    trainingFeatures.add(features2);
	    trainingFeatures.add(features3);
		  trainingFeatures.add(features4);
		
		  trainingLabels.add(label1);
		  trainingLabels.add(label2);
		  trainingLabels.add(label3);
		  trainingLabels.add(label4);
		}
		
//		trainingFeatures.add(features1);
//    trainingFeatures.add(features2);
//    trainingFeatures.add(features3);
//	
//	  trainingLabels.add(label1);
//	  trainingLabels.add(label2);
//	  trainingLabels.add(label3);
		
		Collections.shuffle(trainingFeatures);
		Collections.shuffle(trainingLabels);
		
		smallDataSet.add(trainingFeatures);
		smallDataSet.add(trainingLabels);
		
		return smallDataSet;
	}
	
	// TODO
	private List<List<List<Double>>> testNN(NNManager networkManager, List<Double> storedStableWeights, int runs,
		int cycles, BasicMLDataSet trainingExamples, List<List<Double>> features, List<List<Double>> labels, 
		double lambda, boolean regularise) {
	
		  List<List<List<Double>>> allOutputs = new ArrayList<List<List<Double>>>();
//		  List<Double> allTrainingErrors = new ArrayList<Double>();
//		  List<Double> allClassificationErrors = new ArrayList<Double>();
//		  for (int i = 0; i < runs; i++) {
//		  	List<List<Double>> outputAfterTraining = new ArrayList<List<Double>>(); 
//		    networkManager.initWeightsFromList(storedStableWeights); 
//		    double alpha = 0.03;   
//		    double[] trainingError = null;
//		    if (regularise == true) {
////	        trainingError = networkManager.trainNetwork(cycles, alpha, lambda);
//	        trainingError = null; //networkManager.trainNetwork(cycles, alpha, lambda, true, true);
//		    }
//		    if (regularise == false) {
//		    	trainingError = null; //networkManager.trainNetwork(cycles, alpha, lambda, true, false);
//		    }
//	      allTrainingErrors.add(trainingError[1]);
//	      double classificationError = 0; // was: networkManager.calculateClassificationError(features, labels);
//        allClassificationErrors.add(classificationError);
//	      //	      System.out.println("Training error: " + trainingError[1]);
////	      System.out.println("Classification error 1: " + classificationError);    
//	          
//	      // 3. Evaluate the network 
//	      for (int j = 0; j < features.size(); j++) {
//	        double[] currentLabel = networkManager.evalNetwork(features.get(j));
//	        // Turn currentLabel into a list
//	        List<Double> currentLabelAsList = new ArrayList<Double>();
//	        for (int k = 0; k < currentLabel.length; k++) {
//	      	  currentLabelAsList.add(currentLabel[k]);
//	        }
//	   
//	        // Add the list to outputAfterTraining  
//	        outputAfterTraining.add(currentLabelAsList);
//	        
//	      }
//	      allOutputs.add(outputAfterTraining);
//		  }
//		  System.out.println("\nDimensions of allOutputs (number of runs, number of training examples, size of labels):");
//		  System.out.println(allOutputs.size());
//	    System.out.println(allOutputs.get(0).size());
//	    System.out.println(allOutputs.get(0).get(0).size());
//		  
//	    // TRAINING ERRORS
//	    // Quick visual test
//	    System.out.println("\nallTrainingErrors:");
//	    for(int i = 0; i < allTrainingErrors.size(); i++) {
//	      System.out.println(allTrainingErrors.get(i));
//	    }
//	    
//	    // Assert equality
//	    for(int i = 0; i < allTrainingErrors.size(); i++) {
//	      double reference = allTrainingErrors.get(0);
//	      assertEquals(reference, allTrainingErrors.get(i));
//	    }
//	    
//	    // CLASSIFICATION ERRORS
//	    // Quick visual test
//	    System.out.println("\nallClassificationErrors:");
//	    for(int i = 0; i < allClassificationErrors.size(); i++) {
//	      System.out.println(allClassificationErrors.get(i));
//	    }
//	    
//	    // Assert equality
//	    for(int i = 0; i < allClassificationErrors.size(); i++) {
//	      double reference = allClassificationErrors.get(0);
//	      assertEquals(reference, allClassificationErrors.get(i));
//	    }
//	    
//	    // SUMMED OUTPUTS
//	    // Quick visual test: make sure that each run's output summed is the same 
//	    summedOutputs = new ArrayList<Double>(); 
//	    for (int k = 0; k < runs; k++) {
//	      double sum = 0;
//	    	for (int i = 0; i < features.size(); i++) {
//	  	    for (int j = 0; j < labels.get(0).size(); j++) {
//	  	      sum += allOutputs.get(k).get(i).get(j);
//	  	    }  	
//	  	  }
//	    	summedOutputs.add(sum);
//	    }
//	    System.out.println("\nsummedOutputs");
//	    for (int i = 0; i < summedOutputs.size(); i++) {
//	      System.out.println(summedOutputs.get(i));
//	    }
//	    
//	    // Assert equality
//	    for(int i = 0; i < summedOutputs.size(); i++) {
//	      double reference = summedOutputs.get(0);
//	      assertEquals(reference, summedOutputs.get(i));
//	    }
//	    
//	    // OUTPUTS
//	    // Assert equality     
//	    for (int k = 0; k < runs; k++) {
//	      for (int i = 0; i < features.size(); i++) {
//	      	for (int j = 0; j < labels.get(0).size(); j++) {
//		      	double outputOfRun1AsReference = allOutputs.get(0).get(i).get(j); 
//	  	    	assertEquals(outputOfRun1AsReference, allOutputs.get(k).get(i).get(j));
//		      }
//		    }
//	    }
	    return allOutputs;
  }
	
	// TODO
	private void testRelativeTraining() {
//		ActivationSigmoid sig = new ActivationSigmoid();
//		double in1x=1 , in2x=-2 , in1y=1.5, in2y=1;
//		double w11=-0.5 , w21=0.25;
//		double eps = 0.01;
//		double lrate = 0.5;
//		int cycles = 1;
//		double lambda = 0.0;
//		double weightsInitialization = 1.0; // Initialization from a list
//		
//		// 1-computation of the right values for the Training
//		
//		// computation by hand
//		double netinpx = in1x*w11 + in2x*w21;
//		double netinpy = in1y*w11 + in2y*w21;
//		double outx = sigmoid(netinpx );
//		double outy = sigmoid(netinpy);
//
//		double err= outx>outy+eps? 0: outy-outx+eps;
//		double trgx = outx + err; 
//		double trgy = outy - err; 
//		double d_outx = (outx - trgx)* sig.derivativeFunction( netinpx, outx);
//		double d_outy = (outy - trgy)* sig.derivativeFunction( netinpy, outy);
//
//		double dw11x = d_outx * in1x;
//		double dw21x = d_outx * in2x;
//		double dw11y = d_outy * in1y;
//		double dw21y = d_outy * in2y;	
//
//		double w1[] = new double[]{w11 - (lrate * (dw11x + dw11y)), w21 - (lrate * (dw21x + dw21y))}; 
//		double outxHand = sigmoid(in1x*w1[0] + in2x*w1[1]);
//		double outyHand = sigmoid(in1y*w1[0] + in2y*w1[1]);
//
//		netinpx = in1x*w11 + in2x*w21;
//		netinpy = in1y*w11 + in2y*w21;
//		double outx2 = sigmoid(netinpx );
//		double outy2 = sigmoid(netinpy);
//
//		// 2-computation of the values by the relative training method
//		
//		// creation of a neural network
//		BasicNetwork networkBase = new BasicNetwork();
//		networkBase.addLayer( new BasicLayer( null, false, 2));	
//		networkBase.addLayer( new BasicLayer( sig, false, 1));
//		networkBase.getStructure().finalizeStructure();
//		networkBase.reset();
//		networkBase.setWeight(0, 0, 0, w11);
//		networkBase.setWeight(0, 1, 0, w21);
//
//		EncogNNManager netMan = new EncogNNManager( networkBase );
//		
//		List<Double> wList = Arrays.asList( w11 , w21 );
//		netMan.setWeightsList( wList );
//		
//		// creation of the relative training example
//		List<Double> better = Arrays.asList(in1x,in2x);
//		List<Double> worse = Arrays.asList(in1y,in2y);
//		RelativeTrainingExample relEx = new RelativeTrainingExample( better , worse );
////		List<RelativeTrainingExample> relList = Arrays.asList(relEx);
////		netMan.setRelativeTrainingExamples(Arrays.asList(relEx));
//		
//		
//		// jUnit testing : comparison of the initial results
//		double[] initOutx = netMan.evalNetwork( better );
//		double[] initOuty = netMan.evalNetwork( worse );
//		assertEquals(outx, initOutx[0], 0);
//		assertEquals(outy, initOuty[0], 0);
//
//		// training of the network and getting of the results
//		//		netMan.trainRelative(relList, cycles, lrate, lambda, eps, weightsInitialization);
////		netMan.trainNetworkRelative(cycles, lrate, lambda, eps, false, false,""); // TODO or fixFlatSpot = false 
//		double[] outxRT = netMan.evalNetwork( better );
//		double[] outyRT = netMan.evalNetwork( worse );
//		
//		// jUnit testing : compare the relative method results and the right results calculated by hand
//		System.out.println (" outxHand = " + outxHand);
//		System.out.println (" outxRM = " + outxRT[0]);
//		System.out.println (" initOutx = " + initOutx[0]);
//		System.out.println (" outyHand = " + outyHand);
//		System.out.println (" outyRM = " + outyRT[0]);
//		System.out.println (" initOuty = " + initOuty[0]);
//		
//		assertTrue( initOutx[0] < outxRT[0] );
//		assertTrue( initOuty[0] > outyRT[0] );
//		assertEquals( outxHand, outxRT[0], 0 );
//		assertEquals( outyHand, outyRT[0], 0 );
	}
	
	/** standard sigmoid f(x) = 1/(1 + exp(-x)) */
	// TODO
	private double sigmoid(double x){
		return 1/(1+ Math.exp(-x));
	}
	
}
