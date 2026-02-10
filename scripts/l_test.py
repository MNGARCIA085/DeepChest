from deep_chest.infra.utils import load_leaderboard, update_leaderboard, get_best_run_id, get_top_k_run_ids


import mlflow

def main():
	#load_leaderboard()

	#update_leaderboard('123', '0.5')
	#update_leaderboard('124', '0.9')
	#update_leaderboard('125', '0.7')

	a = get_best_run_id()
	print(a)

	b = get_top_k_run_ids(k=2)
	print(b)


	from deep_chest.infra.tracking import download_best_model, download_all_artifacts
	


	arts = download_all_artifacts(a)
	print(arts)

	from keras.models import load_model
	
	model = load_model(f"{arts}/model/model.keras", compile=False)
	print(model.summary())






if __name__==main():
	main()