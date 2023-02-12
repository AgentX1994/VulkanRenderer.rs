use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::renderer::error::RendererResult;
use crate::renderer::model::Model;
use crate::renderer::vertex::Vertex;
use crate::renderer::InstanceData;

// Helper function to get an index from a vector
fn get_from_vector<R: Copy>(vector: &Vec<R>, index: i64) -> Option<R> {
    match index.cmp(&0) {
        std::cmp::Ordering::Less => {
            let index = vector.len() - index.unsigned_abs() as usize;
            Some(vector[index])
        }
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Greater => Some(vector[index as usize - 1]),
    }
}

fn insert_or_get_index_of_vertex(
    vertex_to_index_map: &mut HashMap<Vertex, u32>,
    vertex_list: &mut Vec<Vertex>,
    vertex: &Vertex,
) -> u32 {
    if let Some(index) = vertex_to_index_map.get(vertex) {
        *index
    } else {
        let index = vertex_list.len() as u32;
        vertex_list.push(*vertex);
        if vertex_to_index_map.insert(*vertex, index).is_some() {
            panic!("Vertex was inserted into map behind our back?!");
        }
        index
    }
}

pub fn load_obj<P: AsRef<Path>>(path: P) -> RendererResult<Model<Vertex, InstanceData>> {
    let obj_file = File::open(path)?;
    let reader = BufReader::new(obj_file);

    // Need separate vectors for vertices, normals, and uvs in order
    // to be able to look them up by index
    let mut vertices = Vec::<[f32; 3]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut uvs = Vec::<[f32; 2]>::new();

    // These are the actual vectors we'll pass into the model
    let mut vertices_structs = Vec::<Vertex>::new();
    let mut indices = Vec::<u32>::new();

    // For deduplicating vertices, we'll need to be able to look up
    // vertices by value. To speed this up, we'll use this hashmap
    // from Vertex to index in the above vertices struct
    let mut vertex_to_index_map = HashMap::<Vertex, u32>::new();
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() || parts[0] == "#" {
            continue;
        }
        let element_type = parts[0];
        let args = &parts[1..];
        match element_type {
            "v" => {
                // handle vertex
                if args.len() != 3 && args.len() != 4 {
                    panic!("Invalid line in obj: {line}")
                }
                if args.len() == 4 {
                    println!("Warning: w argument to v not supported.")
                }
                let x = args[0].parse::<f32>()?;
                let y = args[1].parse::<f32>()?;
                let z = args[2].parse::<f32>()?;
                vertices.push([x, y, z])
            }
            "vt" => {
                // handle tex coords
                if args.len() != 2 && args.len() != 3 {
                    panic!("Invalid line in obj: {line}")
                }
                if args.len() == 3 {
                    println!("Warning: w argument to vt not supported.")
                }
                let u = args[0].parse::<f32>()?;
                let v = args[1].parse::<f32>()?;
                uvs.push([u, v])
            }
            "vn" => {
                // handle normal
                if args.len() != 3 {
                    panic!("Invalid line in obj: {line}")
                }
                let x = args[0].parse::<f32>()?;
                let y = args[1].parse::<f32>()?;
                let z = args[2].parse::<f32>()?;
                normals.push([x, y, z])
            }
            "f" => {
                // handle tri
                if args.len() != 3 {
                    panic!("Non-triangular faces not supported: {line}")
                }
                let vert1 = args[0];
                let vert2 = args[1];
                let vert3 = args[2];

                let mut vertex1 = Vertex::default();
                let mut vertex2 = Vertex::default();
                let mut vertex3 = Vertex::default();
                if vert1.find('/').is_some() {
                    // We are dealing with vertices and uvs and/or normals
                    let vert1_parts: Vec<&str> = vert1.split('/').collect();
                    let vert2_parts: Vec<&str> = vert2.split('/').collect();
                    let vert3_parts: Vec<&str> = vert3.split('/').collect();

                    if vert1_parts.len() != vert2_parts.len()
                        || vert2_parts.len() != vert3_parts.len()
                    {
                        panic!("Invalid line in obj: {line}");
                    }

                    if vert1_parts.len() == 2 {
                        // only vert and uv
                        let v1_index: i64 = vert1_parts[0].parse()?;
                        let v2_index: i64 = vert2_parts[0].parse()?;
                        let v3_index: i64 = vert3_parts[0].parse()?;
                        let u1_index: i64 = vert1_parts[1].parse()?;
                        let u2_index: i64 = vert2_parts[1].parse()?;

                        let v1 = get_from_vector(&vertices, v1_index).expect("Invalid index");
                        let v2 = get_from_vector(&vertices, v2_index).expect("Invalid index");
                        let v3 = get_from_vector(&vertices, v3_index).expect("Invalid index");

                        let u1 = get_from_vector(&uvs, u1_index).expect("Invalid index");
                        let u2 = get_from_vector(&uvs, u2_index).expect("Invalid index");

                        vertex1.pos = v1.into();
                        vertex2.pos = v2.into();
                        vertex3.pos = v3.into();

                        vertex1.uv = u1.into();
                        vertex2.uv = u2.into();
                    } else if vert1_parts[1].is_empty() {
                        // only vert and normal
                        let v1_index: i64 = vert1_parts[0].parse()?;
                        let v2_index: i64 = vert2_parts[0].parse()?;
                        let v3_index: i64 = vert3_parts[0].parse()?;
                        let n1_index: i64 = vert1_parts[2].parse()?;
                        let n2_index: i64 = vert2_parts[2].parse()?;
                        let n3_index: i64 = vert3_parts[2].parse()?;

                        let v1 = get_from_vector(&vertices, v1_index).expect("Invalid index");
                        let v2 = get_from_vector(&vertices, v2_index).expect("Invalid index");
                        let v3 = get_from_vector(&vertices, v3_index).expect("Invalid index");

                        let n1 = get_from_vector(&vertices, n1_index).expect("Invalid index");
                        let n2 = get_from_vector(&vertices, n2_index).expect("Invalid index");
                        let n3 = get_from_vector(&vertices, n3_index).expect("Invalid index");

                        vertex1.pos = v1.into();
                        vertex2.pos = v2.into();
                        vertex3.pos = v3.into();

                        vertex1.normal = n1.into();
                        vertex2.normal = n2.into();
                        vertex3.normal = n3.into();
                    } else {
                        // Full vertex
                        let v1_index: i64 = vert1_parts[0].parse()?;
                        let v2_index: i64 = vert2_parts[0].parse()?;
                        let v3_index: i64 = vert3_parts[0].parse()?;
                        let u1_index: i64 = vert1_parts[1].parse()?;
                        let u2_index: i64 = vert2_parts[1].parse()?;
                        let n1_index: i64 = vert1_parts[2].parse()?;
                        let n2_index: i64 = vert2_parts[2].parse()?;
                        let n3_index: i64 = vert3_parts[2].parse()?;

                        let v1 = get_from_vector(&vertices, v1_index).expect("Invalid index");
                        let v2 = get_from_vector(&vertices, v2_index).expect("Invalid index");
                        let v3 = get_from_vector(&vertices, v3_index).expect("Invalid index");

                        let u1 = get_from_vector(&uvs, u1_index).expect("Invalid index");
                        let u2 = get_from_vector(&uvs, u2_index).expect("Invalid index");

                        let n1 = get_from_vector(&vertices, n1_index).expect("Invalid index");
                        let n2 = get_from_vector(&vertices, n2_index).expect("Invalid index");
                        let n3 = get_from_vector(&vertices, n3_index).expect("Invalid index");

                        vertex1.pos = v1.into();
                        vertex2.pos = v2.into();
                        vertex3.pos = v3.into();

                        vertex1.uv = u1.into();
                        vertex2.uv = u2.into();

                        vertex1.normal = n1.into();
                        vertex2.normal = n2.into();
                        vertex3.normal = n3.into();
                    }
                } else {
                    // Just the vertex indices
                    let v1_index: i64 = vert1.parse()?;
                    let v2_index: i64 = vert2.parse()?;
                    let v3_index: i64 = vert3.parse()?;

                    let v1 = get_from_vector(&vertices, v1_index).expect("Invalid index");
                    let v2 = get_from_vector(&vertices, v2_index).expect("Invalid index");
                    let v3 = get_from_vector(&vertices, v3_index).expect("Invalid index");

                    vertex1.pos = v1.into();
                    vertex2.pos = v2.into();
                    vertex3.pos = v3.into();
                }
                let index1 = insert_or_get_index_of_vertex(
                    &mut vertex_to_index_map,
                    &mut vertices_structs,
                    &vertex1,
                );
                let index2 = insert_or_get_index_of_vertex(
                    &mut vertex_to_index_map,
                    &mut vertices_structs,
                    &vertex2,
                );
                let index3 = insert_or_get_index_of_vertex(
                    &mut vertex_to_index_map,
                    &mut vertices_structs,
                    &vertex3,
                );
                indices.push(index1);
                indices.push(index2);
                indices.push(index3);
            }
            _ => {
                println!("Warning: Unsupported line in obj: {line}")
            }
        }
    }

    Ok(Model::new(vertices_structs, indices))
}
