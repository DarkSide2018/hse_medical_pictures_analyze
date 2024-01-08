DELETE FROM medical_pictures_train mpt
WHERE picture_id IN
    (SELECT picture_id
    FROM
        (SELECT picture_id,
         ROW_NUMBER() OVER( PARTITION BY image_path ORDER BY  picture_id DESC ) AS row_num
        FROM medical_pictures_train mpt  ) t
        WHERE t.row_num > 1);