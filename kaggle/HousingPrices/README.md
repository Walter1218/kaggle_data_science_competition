# Housing Price Task

Base Model: BaseModel.ipynb (Only use numeric features) Score: 0.22155

First improvement Model: BasePlus.py (Using 72 features, here we drop 8 features) Score: 0.13463

Second improvement Model: BasePlus2.py (Add Parameter Fitting function for xgboost training, I try max_depth =3 and 6.

  The results shows as below:

  RMSE: MAX_DEPTH:6: 0.13095

        MAX_DEPTH:3: 0.13031)

4rd improvement Model: BasePlus3.py (RMSE Score:
  
    with No Lasso : thread is 0.7 ,0.13024

    Only Lasso, with no XGboost :
                    thread is 0.7 ,0.13108

    Combined Lasso and Xgboost :(half half)

                    thread is 0.7 ,0.12435;

    )

  5th improvement Model: BasePlusS.py (underworking)
