#include "raylib.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn_layers.h"

#define MAX_POINTS 10000

#define GRID 28
#define CELL 30
#define CANVAS_SIZE (GRID*CELL)

#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 1000

Rectangle canvas = {0, 0, CANVAS_SIZE, CANVAS_SIZE};
Rectangle side   = {CANVAS_SIZE, 0, (SCREEN_WIDTH - CANVAS_SIZE), CANVAS_SIZE};
Rectangle bottom = {0, CANVAS_SIZE, SCREEN_WIDTH, (SCREEN_HEIGHT - CANVAS_SIZE)};

float grid[GRID][GRID] = {0};

float brushSize = 0.8f;


void paintGaussian(int cx, int cy, float sigma){
    int radius = (int)(sigma * 3);

    for(int y = -radius; y <= radius; y++){
        for(int x = -radius; x <= radius; x++){
            int px = cx + x;
            int py = cy + y;

            if (px < 0 || py < 0 || px >= GRID || py >= GRID){
                continue;
            }
            
            float dist = sqrtf(x * x + y * y);

            float value = expf(-(dist*dist)/(2*sigma*sigma));

            grid[py][px] += value * 0.4f;  // increase to make brigther

            if(grid[py][px] > 1){
                grid[py][px] = 1;
            }
        }
    }
}




void LoadData(const char* filepath, Header* h, layer* hidden_layer, layer* output_layer){
    FILE *f = fopen(filepath, "rb");
    if(!f){
        printf("Failed to open %s\n", filepath);
        return;
    }
    

    size_t read = 0;
    read += fread(&h->input, sizeof(int), 1, f);
    read += fread(&h->hidden, sizeof(int), 1, f);
    read += fread(&h->output, sizeof(int), 1, f);
    if(read != 3){
        printf("Header Reading Error\n");
    }

    hidden_layer->in_size = h->input;
    hidden_layer->nr_of_neurons = h->hidden;
    
    output_layer->in_size = hidden_layer->nr_of_neurons;
    output_layer->nr_of_neurons = h->output;
    

    hidden_layer->weights = malloc(sizeof(float) * hidden_layer->in_size * hidden_layer->nr_of_neurons);
    hidden_layer->bias = malloc(sizeof(float) * hidden_layer->nr_of_neurons);
    output_layer->weights = malloc(sizeof(float) * output_layer->in_size * output_layer->nr_of_neurons);
    output_layer->bias = malloc(sizeof(float) * output_layer->nr_of_neurons);
    


    read += fread(hidden_layer->weights, sizeof(float), hidden_layer->in_size * hidden_layer->nr_of_neurons, f);
    read += fread(hidden_layer->bias, sizeof(float), hidden_layer->nr_of_neurons, f);
    read += fread(output_layer->weights, sizeof(float), output_layer->in_size * output_layer->nr_of_neurons, f);
    read += fread(output_layer->bias, sizeof(float), output_layer->nr_of_neurons, f);

    if(read == 0){
        printf("Data Reading Error\n");
    }

    printf("Loaded: input=%d, hidden=%d, output=%d\n", h->input, h->hidden, h->output);
    fclose(f);
}



void CenterGridToInput(float* input){
    // Find center of mass
    float sum = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    
    for(int y = 0; y < GRID; y++){
        for(int x = 0; x < GRID; x++){
            float v = grid[y][x];
            sum += v;
            cx += x * v;
            cy += y * v;
        }
    }
    
    if(sum < 0.01f){
        memset(input, 0, 784 * sizeof(float));
        return;
    }
    
    cx /= sum;
    cy /= sum;
    
    // Calculate shift to center
    int shift_x = (int)(14 - cx);
    int shift_y = (int)(14 - cy);
    
    // Create centered version directly in input
    memset(input, 0, 784 * sizeof(float));
    
    for(int y = 0; y < GRID; y++){
        for(int x = 0; x < GRID; x++){
            int new_x = x + shift_x;
            int new_y = y + shift_y;
            
            if(new_x >= 0 && new_x < GRID && new_y >= 0 && new_y < GRID){
                input[new_y * GRID + new_x] = grid[y][x];
            }
        }
    }
}


int main(){

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "raylib mouse input");

    SetTargetFPS(60);
    SetRandomSeed(13);

    layer hidden_layer;
    layer output_layer;
    Header h;
    LoadData("layers.bin", &h, &hidden_layer, &output_layer);
    
    float* input = malloc(h.input * sizeof(float));
    float* probs = malloc(10 * sizeof(float));
    int prediction = -1;

    while(!WindowShouldClose()){
        
        
        
        // Get the mouse pos and normalize it
        if(IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
            Vector2 mouse = GetMousePosition();
            
            int gx = mouse.x / CELL;
            int gy = mouse.y / CELL;
            
            paintGaussian(gx, gy, brushSize);
            
            // Center and convert to input
            CenterGridToInput(input);
            
            predict_drawing(input, &hidden_layer, &output_layer, probs, &prediction);
        }

        // Reset with C
        if(IsKeyPressed(KEY_C)){
            memset(grid,0,sizeof(grid));
            prediction = -1;
        }

        // Decrease brushSize
        if(IsKeyPressed(KEY_DOWN) && brushSize > 0){
            brushSize -= 0.1f;
        }

        if(IsKeyPressed(KEY_UP)){
            brushSize += 0.1f;
        }


        BeginDrawing();
        
        ClearBackground((Color){18,18,18,255});
        
        DrawRectangleRec(side, LIME);
        DrawRectangleRec(bottom, (Color){12,12,12,255});
        
        // Draw with the brush
        for(int y = 0; y < GRID; y++){
            for(int x = 0; x < GRID; x++){
                float v = grid[y][x];
                Color c = ColorFromNormalized((Vector4){v, v, v, 1});
                DrawRectangle(x * CELL, y * CELL, CELL, CELL, c);
            }
        }


        


        // PRED BOX
        DrawText("PREDICTION", side.x+20, 20, 50, WHITE);

        DrawLine(canvas.width + 20, CELL * 10, SCREEN_WIDTH - 20, CELL * 10, BLACK);

        for(int i = 0; i < 10; i++){
            int y = (CELL * 11) + (i * CELL * 1.7);
            DrawText(TextFormat("%d", i), side.x + 20, y, 20, WHITE);

            DrawRectangle(side.x + 50, y, side.width - 120, 20, (Color){50, 50, 50, 255});

            
            if(prediction != -1){
                DrawRectangle(side.x + 50, y, (side.width - 120) * probs[i], 20, GREEN);
                DrawText(TextFormat("Prob: %.2f", probs[i] * 100), side.x + 50, y + 23, 20, WHITE);
            }
            else{
                DrawText("Prob: 0", side.x + 50, y + 23, 20, WHITE);
            }
        }

        if(prediction == -1){
            DrawText("Draw A digit 0-9", side.x + 20, 140, 40, WHITE);
        }
        else{
            DrawText(TextFormat("%d", prediction), side.x + 125, 90, 200, WHITE);
        }

        // BOTTOM BOX
        DrawText("Left Mouse to Draw", 20, bottom.y + 20, 20, WHITE);
        DrawText("Press C to clear", 20, bottom.y + 50, 20, WHITE);
        DrawText(TextFormat("Brush Size: %f  (Up = +, Down = -)", brushSize), 20, bottom.y + 80, 20, WHITE);


        // Draw Grid
        for(int i=0;i<=GRID;i++){
            DrawLine(i*CELL, 0, i*CELL, CANVAS_SIZE, (Color){40,40,40,255});
            DrawLine(0, i*CELL, CANVAS_SIZE, i*CELL, (Color){40,40,40,255});
        }


        EndDrawing();
    }
    

    free(input);
    free_layer(hidden_layer);
    free_layer(output_layer);
    
    CloseWindow();
    return 0;
}
