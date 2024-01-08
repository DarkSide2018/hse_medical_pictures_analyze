
echo "=========================="
echo "Creating containers"
echo "----------------"

cd local_env

docker container stop medical-postgres
docker container rm medical-postgres

docker run --name medical-postgres-2   \
             -e POSTGRES_USER=hse_medical     \
             -e POSTGRES_PASSWORD=123456    \
             -e POSTGRES_DB=hse_medical -p 5451:5432 \
             postgres


echo "=========================="
echo "local env was created"
echo "----------------"