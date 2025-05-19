#include <iostream>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <functional>
#include <fstream>
#include <filesystem>

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t err = cmd; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " @ " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define MAX_NODES 1024

class GraphletManager {
public:
    GraphletManager(int num_nodes = MAX_NODES, int avg_degree = 4) : num_nodes_(num_nodes) {}

    void generate_random_nodes(const std::string& label, const std::string& column) {
        std::string key = label + "." + column;
        std::vector<int> host_data(num_nodes_);
        for (int i = 0; i < num_nodes_; ++i)
            host_data[i] = 1000 + i;

        int* device_data;
        CUDA_CHECK(cudaMalloc(&device_data, sizeof(int) * num_nodes_));
        CUDA_CHECK(cudaMemcpy(device_data, host_data.data(), sizeof(int) * num_nodes_, cudaMemcpyHostToDevice));

        columns_[key] = device_data;
    }

    void generate_random_edges(const std::string& edge_label, int avg_degree) {
        std::vector<int> offsets(num_nodes_ + 1);
        std::vector<int> values;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, num_nodes_ - 1);

        int edge_cnt = 0;
        for (int i = 0; i < num_nodes_; ++i) {
            offsets[i] = edge_cnt;
            int deg = avg_degree;
            for (int j = 0; j < deg; ++j) {
                values.push_back(dist(rng));
                edge_cnt++;
            }
        }
        offsets[num_nodes_] = edge_cnt;

        int* d_offsets;
        int* d_values;
        CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * offsets.size()));
        CUDA_CHECK(cudaMalloc(&d_values, sizeof(int) * values.size()));
        CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * offsets.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values.data(), sizeof(int) * values.size(), cudaMemcpyHostToDevice));

        edge_csr_[edge_label] = {d_offsets, d_values};
    }

    int* get_column(const std::string& label, const std::string& column) {
        std::string key = label + "." + column;
        if (columns_.count(key)) return columns_[key];

        // Lazy initialization
        generate_random_nodes(label, column);
        return columns_[key];
    }

    void get_csr(const std::string& edge_label, int*& offsets, int*& values) {
        if (edge_csr_.count(edge_label) == 0)
            generate_random_edges(edge_label, 4);  // default degree

        std::tie(offsets, values) = edge_csr_[edge_label];
    }

    int num_nodes() const { return num_nodes_; }

    ~GraphletManager() {
        for (auto& p : columns_)
            cudaFree(p.second);
        for (auto& [label, pair] : edge_csr_) {
            cudaFree(pair.first);
            cudaFree(pair.second);
        }
    }

    void serialize_to_human_readable(const std::string& output_dir) {
        std::filesystem::create_directories(output_dir);

        // Save node columns
        for (const auto& [key, device_ptr] : columns_) {
            std::vector<int> host_data(num_nodes_);
            CUDA_CHECK(cudaMemcpy(host_data.data(), device_ptr, sizeof(int) * num_nodes_, cudaMemcpyDeviceToHost));

            std::ofstream out(output_dir + "/" + key + ".txt");
            out << "# " << key << "\n";
            for (int i = 0; i < num_nodes_; ++i)
                out << i << ": " << host_data[i] << "\n";
            out.close();
        }

        // Save edge CSR lists
        for (const auto& [label, pair] : edge_csr_) {
            int* d_offsets = pair.first;
            int* d_values  = pair.second;

            std::vector<int> h_offsets(num_nodes_ + 1);
            CUDA_CHECK(cudaMemcpy(h_offsets.data(), d_offsets, sizeof(int) * (num_nodes_ + 1), cudaMemcpyDeviceToHost));

            int total_edges = h_offsets.back();
            std::vector<int> h_values(total_edges);
            CUDA_CHECK(cudaMemcpy(h_values.data(), d_values, sizeof(int) * total_edges, cudaMemcpyDeviceToHost));

            std::ofstream out(output_dir + "/" + label + "_csr.txt");
            out << "# CSR for edge label: " << label << "\n";
            for (int i = 0; i < num_nodes_; ++i) {
                out << i << ": ";
                for (int j = h_offsets[i]; j < h_offsets[i + 1]; ++j) {
                    out << h_values[j] << " ";
                }
                out << "\n";
            }
            out.close();
        }

        std::cout << "Human-readable data written to: " << output_dir << "\n";
    }

private:
    int num_nodes_;
    std::unordered_map<std::string, int*> columns_;
    std::unordered_map<std::string, std::pair<int*, int*>> edge_csr_;
};

/**
* MATCH (n:Person) RETURN n.id
*/
__global__ void pipeline1_kernel(int* id_col, int* out_vals, int* out_count, int num_tuples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tuples) return;

    int id_val = id_col[tid];
    int pos = atomicAdd(out_count, 1);
    out_vals[pos] = id_val;
}

void pipeline1(GraphletManager& gm, std::vector<int>& output) {
    int num_tuples = gm.num_nodes();
    int* id_col = gm.get_column("Person", "id");

    int* d_out_vals;
    int* d_out_count;
    CUDA_CHECK(cudaMalloc(&d_out_vals, num_tuples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

    int threadsPerBlock = 256;
    int blocks = (num_tuples + threadsPerBlock - 1) / threadsPerBlock;

    pipeline1_kernel<<<blocks, threadsPerBlock>>>(id_col, d_out_vals, d_out_count, num_tuples);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    output.resize(h_count);
    CUDA_CHECK(cudaMemcpy(output.data(), d_out_vals, h_count * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out_vals));
    CUDA_CHECK(cudaFree(d_out_count));
}

/**
* MATCH (n:Person)-[:knows]->(m:Person) RETURN n.id, m.id
*/
__global__ void pipeline2_kernel(int* person_ids, int* csr_offsets, int* csr_values,
                                 int* out_n_ids, int* out_m_ids, int* out_count, int num_nodes) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= num_nodes) return;

    int n_id = person_ids[n];
    int start = csr_offsets[n];
    int end   = csr_offsets[n + 1];

    for (int i = start; i < end; ++i) {
        int m = csr_values[i];
        int m_id = person_ids[m];
        int pos = atomicAdd(out_count, 1);
        out_n_ids[pos] = n_id;
        out_m_ids[pos] = m_id;
    }
}

void pipeline2(GraphletManager& gm, std::vector<std::pair<int, int>>& output) {
    int num_nodes = gm.num_nodes();
    int* person_ids = gm.get_column("Person", "id");

    int* csr_offsets;
    int* csr_values;
    gm.get_csr("knows", csr_offsets, csr_values);

    int max_output = num_nodes * 8;

    int* d_out_n_ids;
    int* d_out_m_ids;
    int* d_out_count;
    CUDA_CHECK(cudaMalloc(&d_out_n_ids, max_output * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_m_ids, max_output * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

    int threadsPerBlock = 256;
    int blocks = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;

    pipeline2_kernel<<<blocks, threadsPerBlock>>>(person_ids, csr_offsets, csr_values,
                                                  d_out_n_ids, d_out_m_ids, d_out_count, num_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> h_n_ids(h_count), h_m_ids(h_count);
    CUDA_CHECK(cudaMemcpy(h_n_ids.data(), d_out_n_ids, h_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m_ids.data(), d_out_m_ids, h_count * sizeof(int), cudaMemcpyDeviceToHost));

    output.clear();
    for (int i = 0; i < h_count; ++i)
        output.emplace_back(h_n_ids[i], h_m_ids[i]);

    cudaFree(d_out_n_ids);
    cudaFree(d_out_m_ids);
    cudaFree(d_out_count);
}

/*
MATCH (n:Person {id: 1000})-[r:IS_LOCATED_IN]->(p:Place)
		   RETURN
		   	n.firstName AS firstName,
			n.lastName AS lastName,
			n.birthday AS birthday,
			n.locationIP AS locationIP,
			n.browserUsed AS browserUsed,
			p.id AS cityId,
			n.gender AS gender,
			n.creationDate AS creationDate;
*/
struct PersonLocatedInResult {
    int firstName;
    int lastName;
    int birthday;
    int locationIP;
    int browserUsed;
    int cityId;
    int gender;
    int creationDate;
};

__global__ void pipeline3_kernel(
    int* person_ids,
    int* first_names, int* last_names, int* birthdays,
    int* location_ips, int* browsers, int* genders, int* creation_dates,
    int* csr_offsets, int* csr_values, int* place_ids,
    PersonLocatedInResult* out_results,
    int* out_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= MAX_NODES) return;

    if (person_ids[tid] == 1000) {

        int start = csr_offsets[tid];
        int end = csr_offsets[tid + 1];
        
        for (int i = start; i < end; ++i) {
            int place = csr_values[i];
            int pos = atomicAdd(out_count, 1);

            // Store result
            out_results[pos] = {
                first_names[tid], last_names[tid], birthdays[tid],
                location_ips[tid], browsers[tid], place_ids[place],
                genders[tid], creation_dates[tid]
            };
        }
    }
}



void pipeline3(GraphletManager& gm, std::vector<PersonLocatedInResult>& output) {
    int* person_ids = gm.get_column("Person", "id");
    int* first_names = gm.get_column("Person", "firstName");
    int* last_names = gm.get_column("Person", "lastName");
    int* birthdays = gm.get_column("Person", "birthday");
    int* location_ips = gm.get_column("Person", "locationIP");
    int* browsers = gm.get_column("Person", "browserUsed");
    int* genders = gm.get_column("Person", "gender");
    int* creation_dates = gm.get_column("Person", "creationDate");

    int* csr_offsets, *csr_values;
    gm.get_csr("IS_LOCATED_IN", csr_offsets, csr_values);

    int* place_ids = gm.get_column("Place", "id");

    gm.serialize_to_human_readable("./output");

    constexpr int max_results = 32;
    PersonLocatedInResult* d_out_results;
    int* d_out_count;
    CUDA_CHECK(cudaMalloc(&d_out_results, sizeof(PersonLocatedInResult) * max_results));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

    pipeline3_kernel<<<1, 256>>>(person_ids, first_names, last_names, birthdays,
        location_ips, browsers, genders, creation_dates,
        csr_offsets, csr_values, place_ids,
        d_out_results, d_out_count);

    CUDA_CHECK(cudaDeviceSynchronize());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    output.resize(h_count);
    CUDA_CHECK(cudaMemcpy(output.data(), d_out_results, sizeof(PersonLocatedInResult) * h_count, cudaMemcpyDeviceToHost));

    cudaFree(d_out_results);
    cudaFree(d_out_count);
}

/**
MATCH (n:Person {id: 1000 })-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(message:Comment)
		WHERE message.creationDate <= 10000
		RETURN
			friend.id AS personId,
			friend.firstName AS personFirstName,
			friend.lastName AS personLastName,
			message.id AS postOrCommentId,
			message.content AS postOrCommentContent,
			message.creationDate AS postOrCommentCreationDate;
*/

struct FriendCommentResult {
    int personId;
    int personFirstName;
    int personLastName;
    int commentId;
    int commentContent;
    int commentCreationDate;
};

__global__ void pipeline4_kernel(
    int* person_ids,
    int* first_names, int* last_names,
    int* comment_ids, int* comment_contents, int* comment_creation_dates,
    int* knows_offsets, int* knows_values,
    int* has_creator_offsets, int* has_creator_values,
    FriendCommentResult* out_results,
    int* out_count,
    int num_persons,
    int num_comments
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_persons) return;

    // Step 1: Anchor node match
    if (person_ids[tid] != 1000) return;

    int start_knows = knows_offsets[tid];
    int end_knows = knows_offsets[tid + 1];

    for (int i = start_knows; i < end_knows; ++i) {
        int friend_idx = knows_values[i];

        int f_id = person_ids[friend_idx];
        int f_fn = first_names[friend_idx];
        int f_ln = last_names[friend_idx];

        for (int msg = 0; msg < num_comments; ++msg) {
            int start_creator = has_creator_offsets[msg];
            int end_creator   = has_creator_offsets[msg + 1];

            for (int j = start_creator; j < end_creator; ++j) {
                if (has_creator_values[j] == friend_idx) {
                    int ts = comment_creation_dates[msg];
                    if (ts <= 10000) {
                        int pos = atomicAdd(out_count, 1);
                        out_results[pos] = {
                            f_id, f_fn, f_ln,
                            comment_ids[msg],
                            comment_contents[msg],
                            ts
                        };
                    }
                    break;
                }
            }
        }
    }
}


void pipeline4(GraphletManager& gm, std::vector<FriendCommentResult>& output) {
    int num_persons = gm.num_nodes();
    int num_comments = gm.num_nodes(); // assume same size for simplicity

    // Get all columns
    int* person_ids     = gm.get_column("Person", "id");
    int* first_names    = gm.get_column("Person", "firstName");
    int* last_names     = gm.get_column("Person", "lastName");

    int* comment_ids     = gm.get_column("Comment", "id");
    int* contents        = gm.get_column("Comment", "content");
    int* creation_dates  = gm.get_column("Comment", "creationDate");

    int *knows_offsets, *knows_values;
    gm.get_csr("KNOWS", knows_offsets, knows_values);

    int *has_creator_offsets, *has_creator_values;
    gm.get_csr("HAS_CREATOR", has_creator_offsets, has_creator_values);

    // Allocate output buffers
    constexpr int max_output = 1024;
    FriendCommentResult* d_out;
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(FriendCommentResult) * max_output));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int threads = 256;
    int blocks = (num_persons + threads - 1) / threads;

    pipeline4_kernel<<<blocks, threads>>>(
        person_ids, first_names, last_names,
        comment_ids, contents, creation_dates,
        knows_offsets, knows_values,
        has_creator_offsets, has_creator_values,
        d_out, d_count,
        num_persons, num_comments
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    output.resize(h_count);
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, sizeof(FriendCommentResult) * h_count, cudaMemcpyDeviceToHost));

    cudaFree(d_out);
    cudaFree(d_count);
}

/**
MATCH (n:Person {id: 1000 })-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(message:Comment)
		WHERE message.creationDate <= 10000
		RETURN
			friend.id AS personId,
            SUM(message.id AS postOrCommentId)
*/

struct AggEntry {
    int key;
    unsigned long long sum; 
    int hash;
    int valid;
};

__device__ __forceinline__ int hash_int(int x) {
    return x * 2654435761 % 1024;
}

__global__ void pipeline5_kernel(
    int* person_ids,             // Person.id
    int* comment_ids,            // Comment.id
    int* comment_creationDates,  // Comment.creationDate
    int* knows_offsets, int* knows_values,
    int* has_creator_offsets, int* has_creator_values,
    AggEntry* ht,                // Hash table
    int num_persons, int num_comments
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_persons) return;

    if (person_ids[tid] != 1000) return;

    // Step 1: knows
    int begin = knows_offsets[tid];
    int end = knows_offsets[tid + 1];
    for (int i = begin; i < end; ++i) {
        int friend_id = knows_values[i];

        // Step 2: reverse HAS_CREATOR
        for (int msg = 0; msg < num_comments; ++msg) {
            int cstart = has_creator_offsets[msg];
            int cend   = has_creator_offsets[msg + 1];
            for (int j = cstart; j < cend; ++j) {
                if (has_creator_values[j] == friend_id &&
                    comment_creationDates[msg] <= 10000) {

                    // Step 3: insert/update hash table
                    int h = hash_int(friend_id);
                    for (int k = 0; k < 1024; ++k) {
                        int slot = (h + k) % 1024;
                        int old = atomicCAS(&ht[slot].valid, 0, 1);
                        if (old == 0 || ht[slot].key == friend_id) {
                            ht[slot].key = friend_id;
                            atomicAdd(&ht[slot].sum, static_cast<unsigned long long>(comment_ids[msg]));
                            ht[slot].hash = h;
                            break;
                        }
                    }
                }
            }
        }
    }
}

void pipeline5(GraphletManager& gm, std::vector<std::pair<int, long long>>& results) {
    const int table_size = 1024;
    AggEntry* d_table;
    CUDA_CHECK(cudaMalloc(&d_table, sizeof(AggEntry) * table_size));
    CUDA_CHECK(cudaMemset(d_table, 0, sizeof(AggEntry) * table_size));

    int* person_ids = gm.get_column("Person", "id");
    int* comment_ids = gm.get_column("Comment", "id");
    int* creation_dates = gm.get_column("Comment", "creationDate");

    int* knows_off, *knows_val;
    gm.get_csr("KNOWS", knows_off, knows_val);
    int* hc_off, *hc_val;
    gm.get_csr("HAS_CREATOR", hc_off, hc_val);

    int N = gm.num_nodes();

    pipeline5_kernel<<<(N + 255)/256, 256>>>(
        person_ids, comment_ids, creation_dates,
        knows_off, knows_val,
        hc_off, hc_val,
        d_table, N, N
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    std::vector<AggEntry> h_table(table_size);
    CUDA_CHECK(cudaMemcpy(h_table.data(), d_table, sizeof(AggEntry) * table_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_table));

    results.clear();
    for (const auto& e : h_table)
        if (e.valid && e.sum > 0)
            results.emplace_back(e.key, e.sum);
}

using PipelineFn = std::function<void(GraphletManager&)>;
std::unordered_map<std::string, PipelineFn> pipeline_registry;

void register_pipelines() {
    pipeline_registry["pipeline1"] = [](GraphletManager& gm) {
        std::vector<int> results;
        pipeline1(gm, results);
        for (int id : results)
            std::cout << id << " ";
        std::cout << "\nTotal matched: " << results.size() << "\n";
    };

    pipeline_registry["pipeline2"] = [](GraphletManager& gm) {
        std::vector<std::pair<int, int>> results;
        pipeline2(gm, results);
        for (auto& [n, m] : results)
            std::cout << n << " -> " << m << "\n";
        std::cout << "Total matched pairs: " << results.size() << "\n";
    };

    pipeline_registry["pipeline3"] = [](GraphletManager& gm) {
        std::vector<PersonLocatedInResult> results;
        pipeline3(gm, results);
        for (auto& r : results) {
            std::cout << "firstName: " << r.firstName
                    << ", lastName: " << r.lastName
                    << ", birthday: " << r.birthday
                    << ", locationIP: " << r.locationIP
                    << ", browserUsed: " << r.browserUsed
                    << ", cityId: " << r.cityId
                    << ", gender: " << r.gender
                    << ", creationDate: " << r.creationDate
                    << "\n";
        }
        std::cout << "Total matched: " << results.size() << "\n";
    };

    pipeline_registry["pipeline4"] = [](GraphletManager& gm) {
        std::vector<FriendCommentResult> results;
        pipeline4(gm, results);
        for (auto& r : results) {
            std::cout << "personId: " << r.personId
                      << ", firstName: " << r.personFirstName
                      << ", lastName: " << r.personLastName
                      << ", commentId: " << r.commentId
                      << ", content: " << r.commentContent
                      << ", creationDate: " << r.commentCreationDate
                      << "\n";
        }
        std::cout << "Total matched: " << results.size() << "\n";
    };

    pipeline_registry["pipeline5"] = [](GraphletManager& gm) {
        std::vector<std::pair<int, long long>> results;
        pipeline5(gm, results); 

        for (const auto& [personId, idSum] : results) {
            std::cout << "personId: " << personId
                    << ", postOrCommentIdSum: " << idSum
                    << "\n";
        }
        std::cout << "Total unique friends: " << results.size() << "\n";
    };

}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./pipeline_driver <pipeline_name>\n";
        return 1;
    }

    std::string name = argv[1];
    GraphletManager gm;

    register_pipelines();
    auto it = pipeline_registry.find(name);
    if (it == pipeline_registry.end()) {
        std::cerr << "Unknown pipeline: " << name << "\n";
        return 1;
    }

    it->second(gm);
    return 0;
}