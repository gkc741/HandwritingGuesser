#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include "nn_layers.h"





// I want to create a layer with random weights and bias
struct layer create_ran_layer(int nr_of_neurons, int nr_of_inputs){
    struct layer n;
    n.nr_of_neurons = nr_of_neurons;
    n.in_size = nr_of_inputs;

    // i want a random bias for every neuron
    n.bias = malloc(nr_of_neurons * sizeof(float));
    for(int i = 0; i < nr_of_neurons; i++){
        n.bias[i] = ((float)rand() / RAND_MAX) * 0.002 - 0.001;  // random biasa between [-0.001, 0.001]
    }
    
    // He scale 
    float he_scale = sqrt(2.0 / nr_of_inputs);
    // i want in_size weights pr neuron so i need the input size they are going to take
    n.weights = malloc(nr_of_inputs * nr_of_neurons * sizeof(float));
    for(int i = 0; i < nr_of_inputs * nr_of_neurons; i++){
        n.weights[i] = (((float)rand() / RAND_MAX) * 2.0 - 1.0) * he_scale; 
    }
    return n;
}


// Free it ofc
void free_layer(struct layer layer){
    free(layer.bias);
    free(layer.weights);
}



static uint32_t read_be_u32(FILE *f){
    uint8_t bytes[4];
    if(fread(bytes, 1, 4, f) != 4){
        fprintf(stderr, "Failed to read header bytes\n");
        exit(EXIT_FAILURE);
    };
    // CONVERT BIG EDIAN TO SMALL EDIAN
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}


struct dataset read_images(const char* file){
    // in the MNIST dataset the first 4 things are 32bit intigeres in big edian order so i need to handle them seperatley
    // then the rest should be uint_8 so i read them to a very long list of size colm * row 
    // i should use mmap but for now im using fread

    FILE* f = fopen(file, "rb");
    if(!f){
        fprintf(stderr, "Could not open %s\n", file);
        exit(EXIT_FAILURE);
    }

    // READ/PRINT THE HEADER
    uint32_t magic = read_be_u32(f);
    uint32_t images = read_be_u32(f);
    uint32_t rows = read_be_u32(f);
    uint32_t cols = read_be_u32(f);
    printf("magic: %u\nimages: %u\nrows: %u\ncols: %u\n", magic, images, rows, cols);

    // CONSTRUCT THE DATASET
    struct dataset ds = {NULL, rows * cols, NULL, images};
    ds.data = malloc(sizeof(float) * rows * cols * images);
    uint8_t* data = malloc(sizeof(uint8_t) * rows * cols * images);

    // READ VALUES
    uint32_t value = 0;
    value = fread(data, sizeof(uint8_t), rows * cols * images, f);
    printf("read %u points of data\n", value);
    
    if(value != images * rows * cols){
        fprintf(stderr, "Read wrong amount of data\n");
        exit(EXIT_FAILURE);
    }
    
    
    // NORMALIZE THE VALUES
    for(uint32_t i = 0; i < value; i++){
        ds.data[i] = (float)data[i] / 255.0;
    }

    free(data);
    fclose(f);
    return ds;
}

uint8_t* read_labels(const char* file){
    FILE* f = fopen(file, "rb");
    if(!f){
        fprintf(stderr, "Could not open %s\n", file);
        exit(EXIT_FAILURE);
    }

    // READ/PRINT THE HEADER
    uint32_t magic = read_be_u32(f);
    uint32_t images = read_be_u32(f);
    printf("magic: %u\nimages: %u\n", magic, images);

    // INITIALIZE 
    uint8_t* data = malloc(sizeof(uint8_t) * images);
    uint32_t value = 0;

    // READ VALUES
    value = fread(data, sizeof(uint8_t), images, f);
    printf("read %u img labels\n", value);
    
    if(value != images){
        fprintf(stderr, "Read wrong amount of data\n");
        exit(EXIT_FAILURE);
    }

    fclose(f);
    return data;
}


struct dataset read_data(const char* image_file, const char* label_file){
    struct dataset ds = read_images(image_file);
    ds.labels = read_labels(label_file);
    return ds;
}


void free_dataset(struct dataset dataset){
    free(dataset.data);
    free(dataset.labels);
}



// NOW IVE CONSTRUCTED THE LAYERS AND THEN DATA READING TIME FOR THE MATH

inline float relu(float x){  // INLINE MAKES IT SO WHEN I CALL IT IT JUST IMPUTS THE MATH IN WHERE I CALL IT
    return x > 0 ? x : 0;   // IS X > 0 IF YES RETURN X ELSE RETURN 0 (CALLED TENARY OPERATOR)
}


void calc_hidden_layer_output(struct dataset dataset, struct layer hidden_layer, int img_indx, output* input){
    input->size = hidden_layer.nr_of_neurons;
    
    float temp;
    // I NEED TO TIMES THE INPUT BY THE WEIGHTS
    // BUT THE INPUT IS 784 LONG AND THE WEIGHTS IS 784 * NR_OF_NEURONS
    // I NEED FOR EVERY INPUT TIMES IT BY WEIGHTS AND THEN MOVE WEIGHTS 784 WHICH IS THE LENGTH OF ONE NEURONS WEIGHTS
    float* image_data = &dataset.data[img_indx * dataset.data_size];

    #pragma omp parallel for private(temp)  // improveed it from 14 to 12 sec
    for(int j = 0; j < hidden_layer.nr_of_neurons; j++){ // so for every neuron
        temp = hidden_layer.bias[j];
        for(int i = 0; i < hidden_layer.in_size; i++){  // in size should be the dataset row * cols
            // basically for every dataset item for every neuron
            temp += hidden_layer.weights[j * hidden_layer.in_size + i] * image_data[i];
        }
        input->output_list[j] = relu(temp);
    }
}


void calc_output_layer_output(float* output_of_hidden_layer, struct layer output_layer, output* input){
    input->size = output_layer.nr_of_neurons;

    float temp;
    // I NEED TO SAY OUTPUT = INPUT(WHICH IS OUTPUT OF HIDDEN LAYER) @ WEIGHTS + BIAS
    #pragma omp parallel for private(temp)
    for(int i = 0; i < output_layer.nr_of_neurons; i++){
        temp = output_layer.bias[i];
        for(int j = 0; j < output_layer.in_size; j++){
            temp += output_layer.weights[i * output_layer.in_size + j] * output_of_hidden_layer[j];
        }
        input->output_list[i] = temp;
    }
}

// TURN THE OUTPUT FROM OUTPUT LAYER INTO PROBABILITIES USING SOFTMAX
void calc_softmax_prob(output* output_of_output_layer, output* input){
    // output z = {NULL, output_of_output_layer->size};
    // z.output_list = malloc(sizeof(float) * output_of_output_layer->size);
    input->size = output_of_output_layer->size;
    // FIRST WE FIND THE LATRGEST VALUE
    float max = output_of_output_layer->output_list[0];
    for(int i = 0; i < output_of_output_layer->size; i++){
        input->output_list[i] = output_of_output_layer->output_list[i];
        if (input->output_list[i] > max){
            max = input->output_list[i];
        }
    }

    // THEN WE EXPONENTIATE AND MINUS THE MAX VALUE AND SUM IT
    float sum_exp = 0.0;
    for(int i = 0; i < output_of_output_layer->size; i++){
        input->output_list[i] = exp(input->output_list[i] - max);
        sum_exp += input->output_list[i];
    }


    // THEN WE DEVVIDE BY THE SUM TO GET THE PROBABILITY
    for(int i = 0; i < output_of_output_layer->size; i++){
        input->output_list[i] /= sum_exp;
    }

    // return z;    // THIS IS A LIST OF 10 WITH THE PROBABILITY OF EACH BEING THE ONE
}




float calc_loss(dataset dataset, float* predictioin_probabilities, int img_indx){
    int label = dataset.labels[img_indx];
    float loss = -log(predictioin_probabilities[label]);  // -log of the probability it got for the correct digit
    return loss;
}


int argmax(output prob_prediction_list){
    // FIND THE PREDICTID CLASS
    int pred_class = 0;
    float max_prob = prob_prediction_list.output_list[0];
    for(int i = 0; i < prob_prediction_list.size; i++){
        if(prob_prediction_list.output_list[i] > max_prob){
            max_prob = prob_prediction_list.output_list[i];
            pred_class = i;
        }
    }
    return pred_class;
}

void output_layer_errors(dataset dataset, output prob_prediction, int img_indx, float* input){
    int label = dataset.labels[img_indx];

    // float* error_list_of_output = malloc(sizeof(float) * prob_prediction.size);
    
    for(int i = 0; i < prob_prediction.size; i++){
        float t = (i == label) ? 1.0 : 0.0;
        input[i] = prob_prediction.output_list[i] - t;
    }
}


void hidden_layer_errors(output hidden_layer_output, layer output_layer, float* error_list_of_output, float* input){
    int size = hidden_layer_output.size;
    // float* error_list_of_hidden = malloc(sizeof(float) * size);

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(hidden_layer_output.output_list[i] <= 0){
            input[i] = 0;
            continue;
        }
        float sum = 0;
        for(int j = 0; j < output_layer.nr_of_neurons; j++){
            sum += error_list_of_output[j] * output_layer.weights[j * output_layer.in_size + i];
        // BAD SPATIAL LOCALITY BUT WITH EARLY CONTINUE AND SPARSE RELU ITS WORTH
        }
        input[i] = sum;
    }
}




void update_output_weights_and_bias(layer* output_layer, float* error_list_of_output, float eta, output hidden_layer_output){
    // for every neuron (10)
    #pragma omp parallel for
    for(int i = 0; i < output_layer->nr_of_neurons; i++){
        // for every weight of every neuron (100)
        for(int j = 0; j < output_layer->in_size; j++){
            output_layer->weights[i * output_layer->in_size + j] -= eta * error_list_of_output[i] * hidden_layer_output.output_list[j];
        }
        output_layer->bias[i] -= eta * error_list_of_output[i];
    }
    return;
}


void update_hidden_weights_and_bias(layer* hidden_layer, float* error_list_of_hidden, float eta, float* avg_pixl_values){
    #pragma omp parallel for 
    for(int i = 0; i < hidden_layer->nr_of_neurons; i++){
        for(int j = 0; j < hidden_layer->in_size; j++){
            hidden_layer->weights[i * hidden_layer->in_size + j] -= eta * error_list_of_hidden[i] * avg_pixl_values[j];
        }
        hidden_layer->bias[i] -= eta * error_list_of_hidden[i];
    }
    return;
}

void free_workspace(workspace* ws){
    free(ws->hidden_output.output_list);
    free(ws->output_output.output_list);
    free(ws->probs.output_list);
    free(ws->error_output);
    free(ws->error_hidden);
}

float train_NN(dataset dataset, layer* hidden_layer, layer* output_layer, float eta){
    float accuracy = 0.0;
    workspace ws;

    ws.hidden_output.output_list  = malloc(hidden_layer->nr_of_neurons * sizeof(float));
    ws.output_output.output_list  = malloc(output_layer->nr_of_neurons * sizeof(float));
    ws.probs.output_list       = malloc(output_layer->nr_of_neurons * sizeof(float));
    ws.error_output  = malloc(output_layer->nr_of_neurons * sizeof(float));
    ws.error_hidden  = malloc(hidden_layer->nr_of_neurons * sizeof(float));
    /*
    To complete this ill need to change the structure of all of the functions
    calc_hidden_layer_output, calc_output_layer_output, calc_softmax_prob
    output_layer_errors, hidden_layer_errors
    */

    for(int i = 0; i < dataset.nr_of_images; i++){
        // FORWARD
        calc_hidden_layer_output(dataset, *hidden_layer, i, &ws.hidden_output);
        calc_output_layer_output(ws.hidden_output.output_list, *output_layer, &ws.output_output);
        calc_softmax_prob(&ws.output_output, &ws.probs);
        int prediction = argmax(ws.probs);

        // output hidden_output = calc_hidden_layer_output(dataset, *hidden_layer, i);
        // output output_output = calc_output_layer_output(hidden_output.output_list, *output_layer);
        // output probs = calc_softmax_prob(&output_output);

        // BACKWARD
        // CALC ERRORS
        output_layer_errors(dataset, ws.probs, i, ws.error_output);
        hidden_layer_errors(ws.hidden_output, *output_layer, ws.error_output, ws.error_hidden);


        // UPDATE EVERY TIME (BATCH TRAINING WAS SLOWER AND WORSE)
        float* image_data = &dataset.data[i * dataset.data_size];
        update_hidden_weights_and_bias(hidden_layer, ws.error_hidden, eta, image_data);
        update_output_weights_and_bias(output_layer, ws.error_output, eta, ws.hidden_output);


        if(prediction == dataset.labels[i]) accuracy++;
    }
    free_workspace(&ws);
    accuracy = (100.0 * accuracy) / dataset.nr_of_images;
    return accuracy;
}

float test_on_data(dataset dataset, layer* hidden_layer, layer* output_layer){
    float accuracy = 0.0;

    test_workspace tws;
    tws.hidden_output.output_list  = malloc(hidden_layer->nr_of_neurons * sizeof(float));
    tws.output_output.output_list  = malloc(output_layer->nr_of_neurons * sizeof(float));

    for(int i = 0; i < dataset.nr_of_images; i++){
        calc_hidden_layer_output(dataset, *hidden_layer, i, &tws.hidden_output);
        calc_output_layer_output(tws.hidden_output.output_list, *output_layer, &tws.output_output);
        // NO NEED FOR SOFTMAX
        int prediction = argmax(tws.output_output);
        if(prediction == dataset.labels[i]) accuracy++;
        
    }
    free(tws.hidden_output.output_list);
    free(tws.output_output.output_list);
    accuracy =(100.0 * accuracy) / dataset.nr_of_images;
    return accuracy;
}

void save_data(dataset ds, layer* hidden_layer, layer* output_layer){
    Header h = {ds.data_size, hidden_layer->nr_of_neurons, output_layer->nr_of_neurons};
    FILE *f = fopen("layers.bin", "wb");
    fwrite(&h, sizeof(h), 1, f);
    fwrite(hidden_layer->weights, sizeof(float), hidden_layer->in_size * hidden_layer->nr_of_neurons, f);
    fwrite(hidden_layer->bias, sizeof(float), hidden_layer->nr_of_neurons, f);

    fwrite(output_layer->weights, sizeof(float), output_layer->in_size * output_layer->nr_of_neurons, f);
    fwrite(output_layer->bias, sizeof(float), output_layer->nr_of_neurons, f);
    
    fclose(f);
    printf("Saved Data to layers.bin\n");
}

void predict_drawing(float* input, layer* hidden_layer, layer* output_layer, float* out, int* digit){
    // Allocate temporary outputs
    output hidden_out = {0};
    output output_out = {0};
    
    hidden_out.size = hidden_layer->nr_of_neurons;
    output_out.size = output_layer->nr_of_neurons;
    
    hidden_out.output_list = malloc(hidden_layer->nr_of_neurons * sizeof(float));
    output_out.output_list = malloc(output_layer->nr_of_neurons * sizeof(float));
    
    // Forward pass with a dummy dataset pointer (only need the input itself)
    // Manually do what calc_hidden_layer_output does
    float temp;
    for(int j = 0; j < hidden_layer->nr_of_neurons; j++){
        temp = hidden_layer->bias[j];
        for(int i = 0; i < hidden_layer->in_size; i++){
            temp += hidden_layer->weights[j * hidden_layer->in_size + i] * input[i];
        }
        hidden_out.output_list[j] = relu(temp);
    }
    
    // Output layer
    for(int i = 0; i < output_layer->nr_of_neurons; i++){
        temp = output_layer->bias[i];
        for(int j = 0; j < output_layer->in_size; j++){
            temp += output_layer->weights[i * output_layer->in_size + j] * hidden_out.output_list[j];
        }
        output_out.output_list[i] = temp;
    }
    
    // Softmax
    calc_softmax_prob(&output_out, &output_out);
    
    // Copy probabilities to output if needed
    if(out){
        for(int i = 0; i < output_layer->nr_of_neurons; i++){
            out[i] = output_out.output_list[i];
        }
    }
    
    // Get prediction
    int pred = argmax(output_out);
    
    if(digit){
        *digit = pred;
    }
    
    free(hidden_out.output_list);
    free(output_out.output_list);
}