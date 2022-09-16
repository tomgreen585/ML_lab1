// single neuron code
// all functions for Assignment 1
// and Ex1 of Assignment 2 implemented

#include <iostream>
#include <fstream> // for logging
#include <math.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <vector>

//This values are for global_search
double min_tot_err = 10000.0;
double w0_min_err=0.0;
double w1_min_err=0.0;
double bias_min_err=0.0;

bool first = true;


//activation function
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

// one point/sample of training set. 
struct train_point_t{
	std::vector<double> inputs;
	double target;
};

//vector of training samples
struct train_set_t{
  std::vector<train_point_t> train_points;	
};

// usually loaded from file, but in this case training set is small
void set_train_set(train_set_t& ts){
	train_point_t p1;
	p1.inputs = {0.0, 0.0}; //initialize vector
	p1.target = 0.0;
	ts.train_points.push_back(p1);
	train_point_t p2;
	p2.inputs = {0.0, 1.0};
	p2.target = 0.0;
	ts.train_points.push_back(p2);
	train_point_t p3;
	p3.inputs = {1.0, 0.0};
	p3.target = 0.0;
	ts.train_points.push_back(p3);
	train_point_t p4;
	p4.inputs = {1.0, 1.0};
	p4.target = 1.0;
	ts.train_points.push_back(p4);
	
}

// prints all training saples/points
void print_set(const train_set_t& ts){
	for(train_point_t tp:ts.train_points){
		std::cout<<" Inputs ";
		for(double in:tp.inputs) std::cout<<in<<" ";
		std::cout <<" Output: "<<tp.target<<" ";
		std::cout<<std::endl;
	}
}

// to help with plotting the search convergence
// vector to store 
std::vector<double> convergence;
void save_vector_to_file(std::vector<double> v){
	std::string file_name;
	std::cout<<" Enter file name for convergence graph:";
	std::cin>>file_name;
	std::ofstream ofs;
    std::cout<<"Saving vector "<<file_name<<" ..."<<std::endl;
    ofs.open(file_name.c_str(), std::ofstream::out);//, std::ofstream::out | std::ofstream::trunc);
	if (ofs.is_open()){
		int count =0 ;
		for(double ve:v){
			ofs<<count<<" "<<ve<<std::endl;
			count++;
		}
		ofs.close();
	}
}


struct Neuron{
	std::vector<double> weights;   // weights
	double bias;
	double z; // before sigmoid
	double y; // outputs
	void init(int nw);
	double forward(std::vector<double> inputs);
	void print_neuron();
};

// prototype
//void draw_output(Neuron& n)

// sets wieghts and biases
void Neuron::init(int n){
	
	bias = 0.0;
	weights.reserve(2); //reserve memory for weights
	weights.emplace_back(0.0); // faster compared with "push_back()"
	weights.emplace_back(0.0);
	std::cout<<" weight size="<<weights.size();
}

// calculates output
double Neuron::forward(std::vector<double> inputs){
	z = bias;
	for (unsigned int i = 0 ; i < weights.size() ; i++){
		z = z + weights[i]*inputs[i];
	}
	// shorter version
//	z = std::inner_product(weights, weights + nw,inputs.begin(),bias);
	y =  sigmoid(z);
	return y;
}


void Neuron::print_neuron(){
    std::cout<<" bias="<<bias;
    std::cout<<" w0="<<weights[0];
    std::cout<<" w1="<<weights[1]<<std::endl;
	
}

// what is the difference now between 
// output and target
double error(Neuron& neuron, double t){
	return (neuron.y -t);
}

// combined squared error for all training samples
double total_error(Neuron& neuron, const train_set_t& ts){
    double tot_error = 0.0;
    for (unsigned int i =0 ; i < ts.train_points.size() ; i++){
		neuron.forward(ts.train_points[i].inputs);
		double err = neuron.y-ts.train_points[i].target;
		tot_error = tot_error + err*err;}
     return tot_error;
}

// enter 
void manual_entry(Neuron& n,train_set_t ts){
	std::cout<<" Enter bias";
	std::cin>>n.bias;
	std::cout<<" Enter weights [0]";
	std::cin>>n.weights[0];
	std::cout<<" Enter weights [1]";
	std::cin>>n.weights[1];
	//total_error(n, ts);
	double tot_error = 0.0;
	for (unsigned int i=0; i<ts.train_points.size();i++){
		n.forward(ts.train_points[1].inputs);
		double e = n.y-ts.train_points[i].target;
		tot_error = tot_error + e*e;
}
}

void global_search(Neuron& neuron,const train_set_t& train_set ){
   
    //Your code here
	for (double bias = -10.0; bias < 10.0; bias = bias + 0.1){
	  for ( double w0 = -10.0; w0 < 10.0; w0 = w0 + 0.1){
	   for ( double w1 = -10.0; w1 < 10.0; w1 = w1 + 0.1){
 	      //neuron.set_neuron({w0,w1}, bias);
 	      neuron.bias = bias;
 	      neuron.weights[0] = w0;
 	      neuron.weights[1] = w1;
 	      
 	      if (total_error(neuron, train_set) < min_tot_err){
			  min_tot_err = total_error(neuron, train_set);
			  w0_min_err = w0 ;
			  w1_min_err = w1;
			  bias_min_err = bias;}
       }
	}
      
 }   
    // and here 
    std::cout<<" min total error="<<min_tot_err;
    std::cout<<" bias="<<bias_min_err;
    std::cout<<" w0="<<w0_min_err;
    std::cout<<" w1="<<w1_min_err;
}

void gradient_search(Neuron& neuron,const train_set_t& train_set){
    double d = 0.01;
    //double db = 00;
    //double dw0 = 0.0;
    //double dw1 = 0.0;
    double learn_rate = 25.5;
    int n_step = 0;
    //the gradient function is dynamic setting adjusting the responce that the error values at the rate its changing
    //in response eg. kp error changes quickly the error adjusts more rapidly. if error changes slowly then the adjustment
    //step = adjustment. when you increase learning rate(constant) that step is multi. if u increase - making step bigger
    //if make step bigger your gonna make it overstep and your going to make it overstep the 0 point in parab. take giant step
    //if you make it big your result is big. adjustable value you have to perfect, slow but accurate, learning rate big and fast but unaccurate
    
    while (n_step<150){
		
		for (unsigned int i = 0 ; i < train_set.train_points.size(); i++){	
			neuron.forward(train_set.train_points[i].inputs);
             double e0 = error(neuron, train_set.train_points[i].target);
             
             neuron.bias+=d;
             neuron.forward(train_set.train_points[i].inputs);
             double e1 = error(neuron, train_set.train_points[i].target);
             double de_db = (e1*e1-e0*e0)/d;
             neuron.bias-=d;
                
             neuron.weights[0]+=d;
             neuron.forward(train_set.train_points[i].inputs);
             e1 = error(neuron, train_set.train_points[i].target);
             double de_dw0 =(e1*e1-e0*e0)/d;
             neuron.weights[0]-=d;
             
             neuron.weights[1]+=d;
             neuron.forward(train_set.train_points[i].inputs);
             e1 = error(neuron, train_set.train_points[i].target);
             double de_dw1 =(e1*e1-e0*e0)/d;
             neuron.weights[1]-=d;
             
             //neuron.forward(train_set.train_points[i].inputs);
             
             //double dw1_db = (e1-e1)/d;
             //neuron.weights[0]+=d;
             
             neuron.bias = neuron.bias - de_db * learn_rate;
             neuron.weights[0] = neuron.weights[0] - de_dw0 * learn_rate;
             neuron.weights[1] = neuron.weights[1] - de_dw1 * learn_rate;
             //double bias3 = neuron.bias - de_dw1 * learn_rate;
                
             double total_err = total_error(neuron,train_set);
             std::cout<<n_step<<" <- n_step || total error ->   "<<total_err<<std::endl;
             
                //double current_tot_err = tot_err + e*e;
			convergence.push_back(total_err);
			std::cout<<"convergence = "<<total_err<<std::endl;
			n_step++;  
		}
	}
}

// draws y(x0,x1)
void draw_output(Neuron& n){
	// save 
	std::ofstream of;
	of.open("outs.txt", std::ofstream::out);
	if (of.is_open()){
	  double dx = 0.01;
 	  for (double x0 = 0.0; x0 < 1.0; x0 = x0 + dx)
	   for (double x1 = 0.0; x1 < 1.0; x1 = x1 + dx){
		    of<<x0<<" "<<x1<<" "<<n.forward({x0,x1})<<std::endl;
	   }
	  of.close(); 
	  system("gnuplot gplot");
    }
}

  
int main(){
	train_set_t train_set;
	set_train_set(train_set);
	print_set(train_set);
	Neuron neuron;
	neuron.init(2);
	// comment/uncomment functions 
	manual_entry(neuron,train_set);
	global_search(neuron,train_set);
   	//gradient_search(neuron,train_set);
   	//save_vector_to_file(convergence);
   	draw_output(neuron); // use if you want
	 
 } 
   

