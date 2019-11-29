import schnetpack as spk 
from schnetpack.datasets import MD17
from torch.optim import Adam
import schnetpack.train as trn
import os,torch,shutil
import numpy as np 


def run(split_path,dataset_path,n_train=None,n_val=None,n_epochs=1000):

	storage_dir="Info"
	if not os.path.exists(storage_dir):
		os.makedirs(storage_dir)

	if os.path.exists(os.path.join(storage_dir,"checkpoints")):
		shutil.rmtree(os.path.join(storage_dir,"checkpoints"))

	if os.path.exists(os.path.join(storage_dir,"log.csv")):
		os.remove(os.path.join(storage_dir,"log.csv"))

	if os.path.exists(os.path.join(storage_dir,"best_model")):
		os.remove(os.path.join(storage_dir,"best_model"))

	data=MD17(dataset_path)

	atoms,properties=data.get_properties(0)

	train,val,test=spk.train_test_split(
		data=data,
		split_file=split_path,
		)
	
	train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True)
	val_loader = spk.AtomsLoader(val, batch_size=100)

	means, stddevs = train_loader.get_statistics(
		spk.datasets.MD17.energy, divide_by_atoms=True
	)

	with open("out.txt","w+") as file:
		file.write("IN MD17_train")

	print('Mean atomization energy / atom:      {:12.4f} [kcal/mol]'.format(means[MD17.energy][0]))
	print('Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]'.format(stddevs[MD17.energy][0]))

	n_features=64
	schnet = spk.representation.SchNet(
		n_atom_basis=n_features,
		n_filters=n_features,
		n_gaussians=25,
		n_interactions=6,
		cutoff=5.,
		cutoff_network=spk.nn.cutoff.CosineCutoff
	)


	energy_model = spk.atomistic.Atomwise(
		n_in=n_features,
		property=MD17.energy,
		mean=means[MD17.energy],
		stddev=stddevs[MD17.energy],
		derivative=MD17.forces,
		negative_dr=True
	)

	model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

	# tradeoff
	rho_tradeoff = 0.1
	optimizer=Adam(model.parameters(),lr=1e-3)

	# loss function
	def loss(batch, result):
		# compute the mean squared error on the energies
		diff_energy = batch[MD17.energy]-result[MD17.energy]
		err_sq_energy = torch.mean(diff_energy ** 2)

		# compute the mean squared error on the forces
		diff_forces = batch[MD17.forces]-result[MD17.forces]
		err_sq_forces = torch.mean(diff_forces ** 2)

		# build the combined loss function
		err_sq = rho_tradeoff*err_sq_energy + (1-rho_tradeoff)*err_sq_forces

		return err_sq


	# set up metrics
	metrics = [
		spk.metrics.MeanAbsoluteError(MD17.energy),
		spk.metrics.MeanAbsoluteError(MD17.forces)
	]

	# construct hooks
	hooks = [
		trn.CSVHook(log_path=storage_dir, metrics=metrics),
		trn.ReduceLROnPlateauHook(
			optimizer,
			patience=150, factor=0.8, min_lr=1e-6,
			stop_after_min=True
		)
	]

	trainer = trn.Trainer(
		model_path=storage_dir,
		model=model,
		hooks=hooks,
		loss_fn=loss,
		optimizer=optimizer,
		train_loader=train_loader,
		validation_loader=val_loader,
	)

	# check if a GPU is available and use a CPU otherwise
	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"

	# determine number of epochs and train
	trainer.train(
		device=device,
		n_epochs=n_epochs 
		)

	os.rename(os.path.join(storage_dir,"best_model"),os.path.join(storage_dir,"model_new"))













