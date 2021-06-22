<h4>Реализованная функциональность</h4>
<ul>
    <li>Фильтрация RR интервалов;</li>
    <li>Обнаружение ковидных аномалий;</li>
</ul> 
<h4>Особенность проекта в следующем:</h4>
<ul>
 <li>Использование статистических метрик RR интервалов;</li>
 <li>Дополнительная разметка только ковид-спаков. Обучение модели UNet на доп. разметке;</li>
 <li>Conv Ensemble с разными окнами;</li>  
 </ul>
<h4>Основной стек технологий:</h4>
<ul>
   <li>Python</li>
	<li>PyTorch</li>
	<li>CatBoost</li>
  <li>scikit-learn</li>
	<li>Django</li>
	<li>PlotLy</li>
 </ul>
<h4>Демо</h4>
<p>Демо сервиса доступно по адресу: http://daniil311.pythonanywhere.com/app1/plot1d/ </p>
Запуск демо

~~~	
cd demo
cd site1/
./manage.py makemigrations
./manage.py migrate
./manage.py runserver 0.0.0.0:8000
~~~

Go to: http://127.0.0.1:8000
<h4>Среда запуска</h4>
<ul>
    <li>развертывание сервиса производится на Ubuntu 18.04;</li>
    <li>требуется установленный Python 3.6 со всеми зависимостями из requirements.txt;</li>
</ul> 


<h4>Установка</h4>

Выполните 
~~~
git clone https://github.com/CLeDoPbIT/Mister-X.git
cd Mister-X
pip install -r requirements.txt
~~~

<h4>Структура проекта</h4>
<ul>
 <li>models: модели, которые использовались в Conv Ensemble и UNet;</li>
 <li>catboost: финальная модель, которая работает с стат. метриками и вероятностями ковид-аномалий прошлых моделей;</li>
 <li>preprocessing: фильтрация, узкая разметка спайков;</li>  
 </ul>

<h4>Разработчики</h4>
<ul>
   <li>Андрей Шилов data scientist https://t.me/a_team2</li>
	<li>Евгений Бурашников data scientist https://t.me/eburashnikov</li>
	<li>Анжела Бурова data scientist https://t.me/angelaburova</li>
	<li>Анастасия Филатова data scientist https://t.me/anfltv</li>
	<li>Даниил Коновалов data scientist https://t.me/dikonovalov</li>

 </ul>

