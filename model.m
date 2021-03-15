% number of training sequences
num_seqs = 60000;

% Create random binary sequences and reshape for network
rng('default');
XTrain = randi([0, 1], [10,num_seqs]);
XTrain = dlarray(XTrain,"CB");

% Encoder and decoder layers
encoderLayers = layerGraph([featureInputLayer(10,"Name","encoder_input")
    fullyConnectedLayer(100,"Name","encoder_fc_1")
    reluLayer("Name","relu")
    fullyConnectedLayer(50,"Name","encoder_fc_2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(20,"Name","encoder_output")]);

decoderLayers = layerGraph([featureInputLayer(10,"Name","decoder_input")
    fullyConnectedLayer(100,"Name","decoder_fc_1")
    reluLayer("Name","relu")
    fullyConnectedLayer(50,"Name","decoder_fc_2")
    sigmoidLayer("Name","sigmoid")
    fullyConnectedLayer(20,"Name","decoder_output_h3")]);
    
encoder = dlnetwork(encoderLayers);
decoder = dlnetwork(decoderLayers);


executionEnvironment = "auto";

%training options for the network
numEpochs = 10;
miniBatchSize = 100;
numIterationsPerEpoch = floor(num_seqs/miniBatchSize);
iteration = 0;

% encoder/decoder gradient & squared gradient for first iteration
encoderAvGrad = [];
encoderSqGrad = [];
decoderAvGrad = [];
decoderSqGrad = [];

% custom values for learning rate, gradient decay & sq gradient decay
learningRate = 5e-3;
gradDecay = 0.90;
sqGradDecay = 0.95;

% Initialize progress graph
plots = "training-progress";
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

% Custom Learning loop using ADAM with graph visualization
start = tic;
for epoch = 1:numEpochs
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XBatch = XTrain(:,idx);
        XBatch = dlarray(single(XBatch),"CB");
        
         if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XBatch = gpuArray(XBatch);           
         end
        
        % Evaluate the model gradients (encoder & decoder) and loss using dlfeval and the
        % modelGradients function.            
        [encoderGrad, decoderGrad, loss] = dlfeval(@modelGradients, encoder, decoder, XBatch);
           
         % Update the network parameters using the Adam optimizer.
        [decoder, decoderAvGrad, decoderSqGrad] = ...
            adamupdate(decoder,decoderGrad, decoderAvGrad, decoderSqGrad, iteration, learningRate,gradDecay,sqGradDecay);
            
        [encoder,  encoderAvGrad,  encoderSqGrad] = ...
            adamupdate(encoder,encoderGrad, encoderAvGrad, encoderSqGrad, iteration, learningRate,gradDecay,sqGradDecay);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end 
end


% Helper that passes batch to sample function, forwards resulting z vector
% into decoder, calculates loss, updates gradients.
function [encoderGrad, decoderGrad,loss] = modelGradients(encoder, decoder, XBatch)

    [z, zMu, zSigma] = sample(encoder, XBatch);
    
    h3 = forward(decoder, z);
    h3 = dlarray(h3, 'CB');
    
    loss = vae_loss(XBatch, h3, zMu, zSigma);
    
    [decoderGrad,encoderGrad] = dlgradient(loss, decoder.Learnables,encoder.Learnables);
end

% Function that forwards batch through encoder,performs sampling and reparameterization trick.
function [z, zMu, zVar] = sample(encoder, XBatch)

    encodedBatch = forward(encoder, XBatch);    
    zMu = encodedBatch(1:2:end,:);
    zVar = encodedBatch(2:2:end,:);
    
    % Perform Reparameterization Trick
    sz = size(zMu);
    epsilon = randn(sz);
    sigma = exp(.5 * zVar);
    z = epsilon .* sigma + zMu;
    z = reshape(z, [1,1,sz]);
    z = squeeze(z);
    z = dlarray(z, 'CB');
end

% The loss function inspired by the Nature paper
function loss = vae_loss(x,h3,z_mu,z_sigma)
    % Reshape x and x_pred so values can be easily extracted
    x = dlarray(x,'CB');
    x = squeeze(x);
    x = extractdata(x);
    x = permute(x,[2,1]);
    
   % h3 = squeeze(h3);
    h3 = extractdata(h3);
    h3 = permute(h3,[2,1]);
   
    p_0 = h3(:,1:2:end);
    p_1 = h3(:,2:2:end);
     
for i = 1:size(h3,1) 
    for j = 1:size(x,2) 
        actual = x(i,j);       
 
        if(actual == 0)
            % calculate p(x_i=0|z) and store
            px(i,j) = exp(p_0(i,j))/(exp(p_0(i,j)) + exp(p_1(i,j)));
        end
        if(actual == 1)
            % calculate p(x_i=1|z)and store
            px(i,j) = exp(p_1(i,j))/(exp(p_0(i,j)) + exp(p_1(i,j)));
        end
    end
end

% Sum of log( p(x|z) ) values
% Negate log 
px_logs = log(px);
px_logs = -px_logs;
px_log_sum = sum(px_logs,2);
px_log_sum = permute(px_log_sum,[2,1]);

kl_loss = -.5 * sum(1 + z_sigma - z_mu.^2 - exp(z_sigma), 1);
loss = mean(px_log_sum + kl_loss);
end
