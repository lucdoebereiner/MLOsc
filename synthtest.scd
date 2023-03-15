s.options.blockSize = 1;
s.boot;
s.quit;

s.volume = -12;


// todo
// ludvig's idea, train "compsitionally" on what i think "goes with" something
// "arbitrary" and aesnthetic intentionality

// record parameter settings with improv partner sound (ludvig) (and other way round) for training
// mark moments that are complimentary or "good" :)
// two registers (low freq and timbre), mean and variance for spectral features

// temporal behavior, sample features at different time points and put them in the network

// connect to spectral power splitter (sps)

// machine learning, but fun

(
~synthBus = Bus.audio(s, 2);
~inBus = Bus.audio(s, 1);
~parameters = [0, 0, 0, 0, 0, 0, 0];
~analysisData = [0, 0, 0, 0, 0, 0];
)

(

x = { arg freq1 = 80, freq2 = 212, freq3 = 1321, d1 = 0.01, d2 = 0.013, a = 2.0, rq = 0.2;
	var fb = LocalIn.ar(2, [0.2,0.3]);
	var del1 = DelayC.ar(fb[1], 0.2, d1.lag2(0.1));
	var del2 = DelayC.ar(fb[0], 0.2, d2.lag2(0.1));
	del1 = ((a.lag2(0.1) * del1) + 0.5).sin;
	del1 = BBandPass.ar(del1, [freq1, freq2, freq3].lag2(0.1), rq).sum * rq.linlin(0,1,1,0.75);
	LocalOut.ar([del1, del2 * (-1)]);
	Out.ar(~synthBus, Limiter.ar([del1, del2]*0.5, 0.5));
}.play;

~randomPars = {
	~parameters[0] = 30.exprand(4000);
	~parameters[1] = 30.exprand(4000);
	~parameters[2] = 30.exprand(4000);
	~parameters[3] = 0.0001.exprand(0.1);
	~parameters[4] = 0.0001.exprand(0.1);
	~parameters[5] = 0.8.exprand(8.0);
	~parameters[6] = 0.03.rrand(1);
	x.set(\freq1, ~parameters[0], \freq2, ~parameters[1], \freq3, ~parameters[2],
		\d1, ~parameters[3], \d2, ~parameters[4],
		\a, ~parameters[5], \rq, ~parameters[6]);
};
)

x.free

(
~analysis = { arg prediction = 0;
	var synth = In.ar(~synthBus.index, 2);
	var in = In.ar(~inBus.index);
	var analysisInput = Select.ar(prediction, [synth.sum, in]);
	var amp = analysisInput.abs.lag2(0.1);
	var analysis = FluidSpectralShape.kr(analysisInput,
		[\centroid, \spread, \skewness, \kurtosis],
		windowSize: 2048);
	var pitch = FluidPitch.kr(analysisInput, [\pitch], windowSize: 2048);
	var fft = FFT(LocalBuf(2048), analysisInput);
	var flux = FFTFlux.kr(fft)  * (amp > 0.001);
	var trig = Impulse.kr(10);
	var analysisMeans = FluidStats.kr(analysis ++ [pitch, flux], 3)[0];
	SendReply.kr(trig: trig, cmdName: '/analysis', values: analysisMeans);
	Out.ar(0, [synth.sum, analysisInput] * 0.3)
}.play(target: x, addAction: \addAfter)
)

~analysis.free


~receiveAnalysis = OSCFunc({ arg msg; ~analysisData = msg[3..]; }, '/analysis');

~receiveAnalysis.free


~nn = NetAddr.new("127.0.0.1", 1337); 

// training


(
~addPoint = {
	var inAndOut = ~analysisData ++ ~parameters;
	inAndOut.postln;
	~nn.sendMsg("/nn/point", *inAndOut)
}
)

~randomPars.value;
~addPoint.value;

l = OSCFunc({ arg msg; msg[1].round(0.00001).postln; }, '/loss');
l.free;

~nn.sendMsg("/nn/train", 15000);


// prediction

~analysis.set(\prediction, 1);
(
p = OSCFunc({ arg msg; ~parameters = msg[1..].postln;
	TwisterDef.setVal(\freq1, ~parameters[0]);
	TwisterDef.setVal(\freq2, ~parameters[1]);
	TwisterDef.setVal(\freq3, ~parameters[2]);
	TwisterDef.setVal(\d1, ~parameters[3]);
	TwisterDef.setVal(\d2, ~parameters[4]);
	TwisterDef.setVal(\a, ~parameters[5]);
	TwisterDef.setVal(\rq, ~parameters[6]);
	x.setn(\freq1, ~parameters[0],
		\freq2, ~parameters[1],
		\freq3, ~parameters[2],
		\d1, ~parameters[3],
		\d2, ~parameters[4],
		\a, ~parameters[5],
		\rq, ~parameters[6]);
}, '/pred');
)
p.free

(
~predict = {
	var data = ~analysisData;
	~nn.sendMsg("/nn/pred", *data)
}
)


//x.setn(\freq1, [121, 121].postln)

~predict.value;

~analysisData


(
~input = {
	Out.ar(~inBus.index, SinOsc.ar(MouseX.kr(50,2000,1)))
}.play
)

(
~input = {
	Out.ar(~inBus.index, LPF.ar(WhiteNoise.ar(), MouseX.kr(50,12000,1)))
}.play
)


(
// external input
~input = {
	Out.ar(~inBus.index, SoundIn.ar(0))
}.play
)

(
// feedback
~input = {
	Out.ar(~inBus.index, InFeedback.ar(0)*1.5)
}.play
)

~input.free

s.meter

// midi control

MIDIIn.connectAll
MIDIClient.init;
m = MIDIOut(0)


(

TwisterDef(\freq1,
	TwisterControl(16, 0, m,
		ControlSpec(30, 4000, \exp), {arg v;
			x.set(\freq1, v);
			~parameters[0] = v;
		}, {
			~predict.value;
		}));

TwisterDef(\freq2,
	TwisterControl(17, 0, m,
		ControlSpec(30, 4000, \exp), {arg v;
			x.set(\freq2, v);
			~parameters[1] = v;
		}));

TwisterDef(\freq3,
	TwisterControl(18, 0, m,
		ControlSpec(30, 4000, \exp), {arg v;
			x.set(\freq3, v);
			~parameters[2] = v;
		}));


TwisterDef(\d1,
	TwisterControl(19, 0, m,
		ControlSpec(0.0001, 0.1, \exp), {arg v;
			x.set(\d1, v);
			~parameters[3] = v;
		}));


TwisterDef(\d2,
	TwisterControl(20, 0, m,
		ControlSpec(0.0001, 0.1, \exp), {arg v;
			x.set(\d2, v);
			~parameters[4] = v;
		}));

TwisterDef(\a,
	TwisterControl(21, 0, m,
		ControlSpec(0.8, 8, \exp), {arg v;
			x.set(\a, v);
			~parameters[5] = v;
		}));


TwisterDef(\rq,
	TwisterControl(22, 0, m,
		ControlSpec(0.03, 1, \exp), {arg v;
			x.set(\rq, v);
			~parameters[6] = v;
		}));

)

// save the model

~nn.sendMsg("/save", "synth35good");






/// try out synth stuff

(
x = { arg freq1 = 80, freq2 = 212, freq3 = 1321, d1 = 0.01, d2 = 0.013, k = 0.2, a = 2.0, rq = 0.2;
	var fb = LocalIn.ar(4, [0.2,0.3,0.1,0.2]);

	var del11 = DelayC.ar(fb[1].neg, 0.2, d1.lag2(0.1));
	var del12 = DelayC.ar((fb[0]*(k.lag2(0.1))) + (fb[3]*(1-k.lag2(0.1))), 0.2, d1.lag2(0.1));
	
	var del21 = DelayC.ar((fb[3]*(k.neg.lag2(0.1))) + (fb[0]*(1+k.lag2(0.1))), 0.2, d1.lag2(0.1));
	var del22 = DelayC.ar(fb[2].neg, 0.2, d1.lag2(0.1));


	del12 = ((a.lag2(0.1) * del12) + 0.5).sin;
	del21 = BBandPass.ar(del21, [freq1, freq2, freq3].lag2(0.1), rq).sum; // * rq.linlin(0,1,1,0.8);
	LocalOut.ar([del11, del12, del21, del22]);
	Out.ar(0, Limiter.ar([del12*0.5, del21]*0.5, 0.5));
}.play;
)

~randomPars.value;
x.set(\k, 0.4)

x.set(\a, 4)
// how to abstract waveguides into pseudo-ugens


x = { arg freq1 = 80, freq2 = 212, freq3 = 1321, d1 = 0.01, d2 = 0.013, k = 0.2, a = 2.0, rq = 0.2;
	var fb = LocalIn.ar(4, [0.2,0.3,0.1,0.2]);

	var del11 = DelayC.ar(fb[1].neg, 0.2, d1.lag2(0.1));
	var del12 = DelayC.ar( ((fb[0]*(k.lag2(0.1))) + (fb[3]*(1-k.lag2(0.1)))).tanh, 0.2, d1.lag2(0.1));
	
	var del21 = DelayC.ar( ((fb[3]*(k.neg.lag2(0.1))) + (fb[0]*(1+k.lag2(0.1)))).tanh, 0.2, d1.lag2(0.1));
	var del22 = DelayC.ar(fb[2].neg, 0.2, d1.lag2(0.1));


	del12 = ((a.lag2(0.1) * del12) + 0.5).sin;
	del21 = BBandPass.ar(del21, [freq1, freq2, freq3].lag2(0.1), rq).sum; // * rq.linlin(0,1,1,0.8);
	LocalOut.ar([del11, del12, del21, del22]);
	Out.ar(~synthBus, Limiter.ar(del21.dup*0.5, 0.5));
}.play;

