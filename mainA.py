from fn.model import Model


model = Model()
model.load_data("./fake_news.csv")

# запускаем дробление данных
x_test, y_test = model.slit_data()

# запускаем обучение

model.train()

# тут меня постигло разочерование, толи модель слишком хороша и обучается с 1 эпохе на 98% толи я чтото делаю не так хотя вроде как бы всё правильно, 
# но адекватно и изящно прогрессию не увидеть


model.test(x_test, y_test)